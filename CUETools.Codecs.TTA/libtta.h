/*
 * libtta.h
 *
 * Description: TTA1-C++ library interface
 * Copyright (c) 1999-2015 Aleksander Djuric. All rights reserved.
 * Distributed under the GNU Lesser General Public License (LGPL).
 * The complete text of the license can be found in the COPYING
 * file included in the distribution.
 *
 */

#ifndef _LIBTTA_H
#define _LIBTTA_H

#ifdef __GNUC__
#include <stdint.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h> 
#include <stdexcept>

#ifdef CARIBBEAN
#define ALLOW_OS_CODE 1

#include "../../../rmdef/rmdef.h"
#include "../../../rmlibcw/include/rmlibcw.h"
#include "../../../rmcore/include/rmcore.h"
#endif
#else // MSVC
#include <windows.h>
#include <stdexcept>
#endif

#define MAX_DEPTH 3
#define MAX_BPS (MAX_DEPTH*8)
#define MIN_BPS 16
#define MAX_NCH 6
#define TTA_FIFO_BUFFER_SIZE 5120

#ifdef __GNUC__
#define CALLBACK
#define TTA_EXTERN_API __attribute__((visibility("default")))
#define TTA_ALIGNED(n) __attribute__((aligned(n), packed))
#define __forceinline static __inline
#else // MSVC
#define CALLBACK __stdcall
#define TTA_EXTERN_API __declspec(dllexport)
#define TTA_ALIGNED(n) __declspec(align(n))
#endif

// portability definitions
#ifdef __GNUC__
#ifdef CARIBBEAN
typedef RMint8 (TTAint8);
typedef RMint16 (TTAint16);
typedef RMint32 (TTAint32);
typedef RMint64 (TTAint64);
typedef RMuint8 (TTAuint8);
typedef RMuint16 (TTAuint16);
typedef RMuint32 (TTAuint32);
typedef RMuint64 (TTAuint64);
#else // GNUC
typedef int8_t (TTAint8);
typedef int16_t (TTAint16);
typedef int32_t (TTAint32);
typedef int64_t (TTAint64);
typedef uint8_t (TTAuint8);
typedef uint16_t (TTAuint16);
typedef uint32_t (TTAuint32);
typedef uint64_t (TTAuint64);
#endif
#else // MSVC
typedef __int8 (TTAint8);
typedef __int16 (TTAint16);
typedef __int32 (TTAint32);
typedef __int64 (TTAint64);
typedef unsigned __int8 (TTAuint8);
typedef unsigned __int16 (TTAuint16);
typedef unsigned __int32 (TTAuint32);
typedef unsigned __int64 (TTAuint64);
#endif

// TTA audio format
#define TTA_FORMAT_SIMPLE 1
#define TTA_FORMAT_ENCRYPTED 2

typedef enum tta_error {
	TTA_NO_ERROR,	// no known errors found
	TTA_OPEN_ERROR,	// can't open file
	TTA_FORMAT_ERROR,	// not compatible file format
	TTA_FILE_ERROR,	// file is corrupted
	TTA_READ_ERROR,	// can't read from input file
	TTA_WRITE_ERROR,	// can't write to output file
	TTA_SEEK_ERROR,	// file seek error
	TTA_MEMORY_ERROR,	// insufficient memory available
	TTA_PASSWORD_ERROR,	// password protected file
	TTA_NOT_SUPPORTED	// unsupported architecture
} TTA_CODEC_STATUS;

typedef enum {
	CPU_ARCH_UNDEFINED,
	CPU_ARCH_IX86_SSE2,
	CPU_ARCH_IX86_SSE3,
	CPU_ARCH_IX86_SSE4_1,
	CPU_ARCH_IX86_SSE4_2
} CPU_ARCH_TYPE;

typedef struct {
	TTAuint32 format;	// audio format
	TTAuint32 nch;	// number of channels
	TTAuint32 bps;	// bits per sample
	TTAuint32 sps;	// samplerate (sps)
	TTAuint32 samples;	// data length in samples
} TTA_ALIGNED(16) TTA_info;

typedef struct {
	TTAint32 index;
	TTAint32 error;
	TTAint32 round;
	TTAint32 shift;
	TTAint32 qm[8];
	TTAint32 dx[24];
	TTAint32 dl[24];
} TTA_ALIGNED(16) TTA_fltst;

typedef struct {
	TTAuint32 k0;
	TTAuint32 k1;
	TTAuint32 sum0;
	TTAuint32 sum1;
} TTA_ALIGNED(16) TTA_adapt;

typedef struct {
	TTA_fltst fst;
	TTA_adapt rice;
	TTAint32 prev;
} TTA_ALIGNED(16) TTA_codec;

typedef struct _tag_TTA_io_callback {
	TTAint32 (CALLBACK *read)(struct _tag_TTA_io_callback *, TTAuint8 *, TTAuint32);
	TTAint32 (CALLBACK *write)(struct _tag_TTA_io_callback *, TTAuint8 *, TTAuint32);
	TTAint64 (CALLBACK *seek)(struct _tag_TTA_io_callback *, TTAint64 offset);
} TTA_ALIGNED(16) TTA_io_callback;

typedef struct {
	TTAuint8 buffer[TTA_FIFO_BUFFER_SIZE];
	TTAuint8 end;
	TTAuint8 *pos;
	TTAuint32 bcount; // count of bits in cache
	TTAuint32 bcache; // bit cache
	TTAuint32 crc;
	TTAuint32 count;
	TTA_io_callback *io;
} TTA_ALIGNED(16) TTA_fifo;

// progress callback
typedef void (CALLBACK *TTA_CALLBACK)(TTAuint32, TTAuint32, TTAuint32);

// architecture type compatibility
TTA_EXTERN_API CPU_ARCH_TYPE tta_binary_version();

namespace tta
{
	/////////////////////// TTA decoder functions /////////////////////////
	class TTA_EXTERN_API tta_decoder {
	public:
		bool seek_allowed;	// seek table flag

		tta_decoder(TTA_io_callback *iocb);
		virtual ~tta_decoder();

		void init_get_info(TTA_info *info, TTAuint64 pos);
		void init_set_info(TTA_info *info);
		void set_password(void const *pstr, TTAuint32 len);
		void frame_reset(TTAuint32 frame, TTA_io_callback *iocb);
		int process_stream(TTAuint8 *output, TTAuint32 out_bytes, TTA_CALLBACK tta_callback=NULL);
		int process_frame(TTAuint32 in_bytes, TTAuint8 *output, TTAuint32 out_bytes);
		void set_position(TTAuint32 seconds, TTAuint32 *new_pos);
		TTAuint32 get_rate();

	protected:
		TTA_codec decoder[MAX_NCH]; // decoder (1 per channel)
		TTAint8 data[8];	// decoder initialization data
		TTA_fifo fifo;
		TTA_codec *decoder_last;
		bool password_set;	// password protection flag
		TTAuint64 *seek_table; // the playing position table
		TTAuint32 format;	// tta data format
		TTAuint32 rate;	// bitrate (kbps)
		TTAuint64 offset;	// data start position (header size, bytes)
		TTAuint32 frames;	// total count of frames
		TTAuint32 depth;	// bytes per sample
		TTAuint32 flen_std;	// default frame length in samples
		TTAuint32 flen_last;	// last frame length in samples
		TTAuint32 flen;	// current frame length in samples
		TTAuint32 fnum;	// currently playing frame index
		TTAuint32 fpos;	// the current position in frame

		bool read_seek_table();
		void frame_init(TTAuint32 frame, bool seek_needed);
	}; // class tta_decoder

	/////////////////////// TTA encoder functions /////////////////////////
	class TTA_EXTERN_API tta_encoder {
	public:
		tta_encoder(TTA_io_callback *iocb);
		virtual ~tta_encoder();

		void init_set_info(TTA_info *info, TTAuint64 pos);
		void set_password(void const *pstr, TTAuint32 len);
		void frame_reset(TTAuint32 frame, TTA_io_callback *iocb);
		void process_stream(TTAuint8 *input, TTAuint32 in_bytes, TTA_CALLBACK tta_callback=NULL);
		void process_frame(TTAuint8 *input, TTAuint32 in_bytes);
		void finalize();
		TTAuint32 get_rate();

	protected:
		TTA_codec encoder[MAX_NCH]; // encoder (1 per channel)
		TTAint8 data[8];	// encoder initialization data
		TTA_fifo fifo;
		TTA_codec *encoder_last;
		TTAuint64 *seek_table; // the playing position table
		TTAuint32 format;	// tta data format
		TTAuint32 rate;	// bitrate (kbps)
		TTAuint64 offset;	// data start position (header size, bytes)
		TTAuint32 frames;	// total count of frames
		TTAuint32 depth;	// bytes per sample
		TTAuint32 flen_std;	// default frame length in samples
		TTAuint32 flen_last;	// last frame length in samples
		TTAuint32 flen;	// current frame length in samples
		TTAuint32 fnum;	// currently playing frame index
		TTAuint32 fpos;	// the current position in frame
		TTAuint32 shift_bits; // packing int to pcm

		void write_seek_table();
		void frame_init(TTAuint32 frame);
	}; // class tta_encoder

	//////////////////////// TTA exception class //////////////////////////
	class tta_exception : public std::exception {
		tta_error err_code;

	public:
		tta_exception(tta_error code) : err_code(code) {}
		tta_error code() const { return err_code; }
	}; // class tta_exception
} // namespace tta

#endif // _LIBTTA_H
