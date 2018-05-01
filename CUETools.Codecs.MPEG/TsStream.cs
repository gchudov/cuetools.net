using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Codecs.MPEG
{
    internal class TsStream
    {
        internal UInt16 channel;                    // channel number (1,2 ...)
        internal int streamId;                      // stream number in channel
        internal byte type;                         // 0xff                 - not ES
        internal AudioPCMConfig pcm;
        internal byte[] savedBuffer;
        internal int savedBufferOffset;
        internal int savedBufferSize;

        internal byte[] psi;                              // PAT,PMT cache (only for PSI streams)
        internal int psi_len;
        internal int psi_offset;
        internal int psi_table;
        internal bool at_packet_header;

        internal byte ts_stream_id;                     // MPEG stream id

        internal bool is_opened;

        internal UInt64 dts;                          // current MPEG stream DTS (presentation time for audio, decode time for video)
        internal UInt64 first_dts;
        internal UInt64 first_pts;
        internal UInt64 last_pts;
        internal UInt32 frame_length;                 // frame length in ticks (90 ticks = 1 ms, 90000/frame_length=fps)
        internal UInt32 frame_size;                   // frame size in bytes
        internal UInt64 frame_num;                    // frame counter

        internal TsStream()
        {
            is_opened = false;
            psi = new byte[512];
            psi_len = 0;
            psi_offset = 0;
            psi_table = 0;
            channel = 0xffff;
            streamId = 0;
            type = 0xff;
            ts_stream_id = 0;
            dts = 0;
            first_dts = 0;
            first_pts = 0;
            last_pts = 0;
            frame_length = 0;
            frame_size = 0;
            frame_num = 0;
            pcm = null;
            savedBuffer = new byte[192];
            savedBufferOffset = 0;
            savedBufferSize = 0;
        }
    };
}
