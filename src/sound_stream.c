#include <string.h>
#include <kernel.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#include <errno.h>
#include <math.h>

#include <sound.h>
#include <taskman.h>
#include <dbgprintf.h>

static int master_volume = 0;

static volatile bool stream_playing = false;

static char soundBuffer[AUDIO_STREAM_BUFFER_SIZE] __attribute__((aligned(64)));

static SoundStream * volatile cur_snd = NULL;

/*
 * Streaming is handled by a dedicated EE thread instead of the audsrv
 * fill-buffer callback. The fill-buffer callback runs in interrupt context,
 * where blocking file I/O (fread) and heavy Vorbis decoding (ov_read, which
 * allocates memory and is not reentrant) are illegal: this crashed PCSX2 for
 * OGG files and produced unstable playback for WAV files. audsrv_play_audio()
 * blocks until the ring buffer has room, so a normal thread paces itself.
 */
static int stream_thread_id = -1;
static int stream_sema_id = -1;

/* Read the next PCM chunk from a WAV file. Returns bytes written to buffer. */
static int fill_wav(SoundStream *snd) {
    uint32_t remaining = snd->data_size - snd->data_read;
    uint32_t want = sizeof(soundBuffer);

    if (remaining == 0)
        return 0;

    if (want > remaining)
        want = remaining;

    int ret = fread(soundBuffer, 1, want, snd->fp);
    if (ret > 0)
        snd->data_read += ret;

    return ret;
}

/* Decode the next PCM chunk from an OGG file. Returns bytes written to buffer. */
static int fill_ogg(SoundStream *snd) {
    int bitStream = 0;
    int bufferPtr = 0;

    while (bufferPtr < AUDIO_STREAM_BUFFER_SIZE) {
        int ret = ov_read(snd->fp, soundBuffer + bufferPtr,
                          AUDIO_STREAM_BUFFER_SIZE - bufferPtr, 0, 2, 1, &bitStream);

        if (ret > 0) {
            bufferPtr += ret;
        } else if (ret == 0) {
            break; /* end of stream */
        } else {
            dbgprintf("ogg: decode error %d.\n", ret);
            break;
        }
    }

    return bufferPtr;
}

static void stream_rewind(SoundStream *snd) {
    if (snd->type == OGG_AUDIO) {
        ov_pcm_seek(snd->fp, 0);
    } else if (snd->type == WAV_AUDIO) {
        fseek(snd->fp, snd->data_start, SEEK_SET);
        snd->data_read = 0;
    }
}

static int stream_thread(void *arg) {
    while (true) {
        WaitSema(stream_sema_id);

        while (stream_playing && cur_snd != NULL) {
            SoundStream *snd = cur_snd;

            int bytes = (snd->type == OGG_AUDIO) ? fill_ogg(snd) : fill_wav(snd);

            if (bytes > 0)
                audsrv_play_audio(soundBuffer, bytes);

            if (bytes < AUDIO_STREAM_BUFFER_SIZE) {
                /* Reached end of stream. */
                stream_rewind(snd);

                if (!snd->loop)
                    stream_playing = false;
            }
        }
    }

    return 0;
}

static void ensure_stream_thread(void) {
    if (stream_thread_id >= 0)
        return;

    ee_sema_t sema;
    sema.init_count = 0;
    sema.max_count = 1;
    sema.option = 0;
    stream_sema_id = CreateSema(&sema);

    stream_thread_id = create_task("Sound: Streaming Thread", (void*)stream_thread, 16384, 40);
    init_task(stream_thread_id, NULL);
}

/*
 * Parse the RIFF/WAVE structure, locating the "fmt " and "data" chunks
 * instead of assuming a fixed 44/48-byte header. This fixes unstable playback
 * for files carrying extra chunks (LIST, fact, JUNK, ...) or a non-canonical
 * fmt chunk, which previously made AthenaEnv read the wrong format and play
 * audio data from the wrong offset.
 */
static bool parse_wav(SoundStream *wav) {
    FILE *fp = wav->fp;
    char id[4];
    uint32_t size;

    fseek(fp, 0, SEEK_SET);

    if (fread(id, 1, 4, fp) != 4 || memcmp(id, "RIFF", 4) != 0)
        return false;

    fseek(fp, 4, SEEK_CUR); /* skip RIFF chunk size */

    if (fread(id, 1, 4, fp) != 4 || memcmp(id, "WAVE", 4) != 0)
        return false;

    bool have_fmt = false;
    bool have_data = false;

    while (fread(id, 1, 4, fp) == 4 && fread(&size, 1, 4, fp) == 4) {
        long chunk_start = ftell(fp);

        if (memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmttag, channels, blockalign, bits;
            uint32_t samplerate, byterate;

            fread(&fmttag, 1, 2, fp);
            fread(&channels, 1, 2, fp);
            fread(&samplerate, 1, 4, fp);
            fread(&byterate, 1, 4, fp);
            fread(&blockalign, 1, 2, fp);
            fread(&bits, 1, 2, fp);

            wav->fmt.channels = channels;
            wav->fmt.freq = samplerate;
            wav->fmt.bits = bits;
            wav->byte_rate = byterate ? byterate : (samplerate * channels * (bits / 8));
            have_fmt = true;
        } else if (memcmp(id, "data", 4) == 0) {
            wav->data_start = (uint32_t)chunk_start;
            wav->data_size = size;
            have_data = true;
            break;
        }

        /* chunks are word-aligned */
        fseek(fp, chunk_start + size + (size & 1), SEEK_SET);
    }

    if (!have_fmt || !have_data)
        return false;

    fseek(fp, wav->data_start, SEEK_SET);
    wav->data_read = 0;
    return true;
}

SoundStream * load_wav(const char* path) {
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        dbgprintf("wav: Failed to open file %s\n", path);
        return NULL;
    }

    SoundStream *wav = calloc(1, sizeof(SoundStream));
    if (wav == NULL) {
        fclose(fp);
        return NULL;
    }

    wav->fp = fp;
    wav->type = WAV_AUDIO;
    wav->loop = false;

    if (!parse_wav(wav)) {
        dbgprintf("wav: Invalid or unsupported WAV file %s\n", path);
        fclose(fp);
        free(wav);
        return NULL;
    }

    return wav;
}

// OGG Support
SoundStream* load_ogg(const char* path) {
    FILE *oggFile = fopen(path, "rb");
    if (oggFile == NULL) {
        dbgprintf("ogg: Failed to open Ogg file %s\n", path);
        return NULL;
    }

    SoundStream* ogg = calloc(1, sizeof(SoundStream));
    if (ogg == NULL) {
        fclose(oggFile);
        return NULL;
    }

    ogg->fp = calloc(1, sizeof(OggVorbis_File));
    if (ogg->fp == NULL) {
        fclose(oggFile);
        free(ogg);
        return NULL;
    }

    if (ov_open_callbacks(oggFile, ogg->fp, NULL, 0, OV_CALLBACKS_DEFAULT) < 0) {
        dbgprintf("ogg: Input does not appear to be an Ogg bitstream.\n");
        fclose(oggFile);
        free(ogg->fp);
        free(ogg);
        return NULL;
    }

	vorbis_info *vi = ov_info(ogg->fp, -1);
    ov_pcm_seek(ogg->fp, 0);

    ogg->fmt.channels = vi->channels;
    ogg->fmt.freq = vi->rate;
    ogg->fmt.bits = 16;
    ogg->type = OGG_AUDIO;
    ogg->loop = false;

    return ogg;
}

SoundStream *sound_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        dbgprintf("sound: Failed to open file %s\n", path);
        return NULL;
    }

	uint32_t magic = 0;
	fread(&magic, 1, 4, f);
	fclose(f);

	switch (magic) {
		case 0x5367674F: /* OGG */
			return load_ogg(path);
		case 0x46464952: /* WAV */
			return load_wav(path);
	}

    dbgprintf("sound: Unsupported file format (magic 0x%08X)\n", magic);
    return NULL;
}

void sound_play(SoundStream * snd) {
    if (snd == NULL)
        return;

    if (stream_playing)
        return;

    ensure_stream_thread();

    cur_snd = snd;

    audsrv_set_format(&(cur_snd->fmt));
    audsrv_set_volume(master_volume);

    stream_playing = true;
    SignalSema(stream_sema_id);
}

int is_sound_playing(SoundStream* snd) {
    return ((snd == cur_snd) && stream_playing);
}

void sound_pause() {
	if (stream_playing) {
		stream_playing = false;
        /* Unblock the streaming thread if it is inside audsrv_play_audio(). */
        audsrv_stop_audio();
	}
}

void sound_free(SoundStream* snd) {
    if (snd == NULL)
        return;

    if (snd == cur_snd) {
        sound_pause();
        cur_snd = NULL;
    }

    if (snd->type == OGG_AUDIO) {
        ov_clear(snd->fp);
        free(snd->fp);
    } else if (snd->type == WAV_AUDIO) {
        fclose(snd->fp);
    }

    snd->fp = NULL;
    free(snd);
}

void sound_setvolume(int volume) {
	audsrv_set_volume(volume);
    master_volume = volume;
}

void sound_rewind(SoundStream* snd) {
    if (snd == NULL)
        return;

    stream_rewind(snd);
}

int sound_get_duration(SoundStream* snd) {
    if (snd == NULL)
        return -1;

    if (snd->type == OGG_AUDIO) {
        return (int)(ov_time_total(snd->fp, -1) * 1000);
    } else if (snd->type == WAV_AUDIO) {
        if (snd->byte_rate == 0)
            return -1;

        return (int)(((uint64_t)snd->data_size * 1000) / snd->byte_rate);
    }

    return -1;
}

void sound_set_position(SoundStream* snd, int ms) {
    if (snd == NULL || ms < 0 || ms >= sound_get_duration(snd))
        return;

    bool was_current = (snd == cur_snd);
    if (was_current)
        sound_pause();

    if (snd->type == OGG_AUDIO) {
        ov_time_seek(snd->fp, ms / 1000.0);
    } else if (snd->type == WAV_AUDIO) {
        uint32_t offset = (uint32_t)(((uint64_t)ms * snd->byte_rate) / 1000);

        if (snd->fmt.channels && snd->fmt.bits) {
            uint32_t align = snd->fmt.channels * (snd->fmt.bits / 8);
            if (align)
                offset -= (offset % align);
        }

        if (offset > snd->data_size)
            offset = snd->data_size;

        fseek(snd->fp, snd->data_start + offset, SEEK_SET);
        snd->data_read = offset;
    }

    if (was_current)
        sound_play(snd);
}

int sound_get_position(SoundStream* snd) {
    if (snd == NULL)
        return -1;

    if (snd->type == OGG_AUDIO) {
        return (int)(ov_time_tell(snd->fp) * 1000);
    } else if (snd->type == WAV_AUDIO) {
        if (snd->byte_rate == 0)
            return -1;

        return (int)(((uint64_t)snd->data_read * 1000) / snd->byte_rate);
    }

    return -1;
}
