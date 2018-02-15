using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace CUETools.Codecs.BDLPCM
{
    [AudioDecoderClass("cuetools", "mpls", 2)]
    public class MPLSReader : IAudioSource
    {
        public unsafe MPLSReader(string path, Stream IO)
        {
            settings = new BDLPCMReaderSettings();
            _path = path;
            _IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000);
            int length = (int)_IO.Length;
            contents = new byte[length];
            if (_IO.Read(contents, 0, length) != length) throw new Exception("");
            fixed (byte* ptr = &contents[0])
            {
                FrameReader fr = new FrameReader(ptr, length);
                hdr_m = parseHeader(fr);
                fr = new FrameReader(ptr + hdr_m.list_pos, length - hdr_m.list_pos);
                parsePlaylist(fr);
                fr = new FrameReader(ptr + hdr_m.mark_pos, length - hdr_m.mark_pos);
                parsePlaylistMarks(fr);
            }
        }

        void openEntries()
        {
            readers = new List<BDLPCMReader>();
            var pids = new List<ushort>();
            foreach (var item in hdr_m.play_item)
                foreach (var audio in item.audio)
                {
                    if (audio.coding_type != 0x80 /* LPCM */) continue;
                    pids.Add(audio.pid);
                }
            ushort chosenPid;
            if (settings.Pid.HasValue)
            {
                if (!pids.Contains(settings.Pid.Value))
                    throw new Exception("Pid can be " +
                        string.Join(", ", pids.ConvertAll(pid => pid.ToString()).ToArray()));
                chosenPid = settings.Pid.Value;
            }
            else if (settings.Stream.HasValue)
            {
                if (settings.Stream.Value < 0 || settings.Stream.Value >= pids.Count)
                    throw new Exception("Stream can be 0.." + (pids.Count - 1).ToString());
                chosenPid = pids[settings.Stream.Value];
            }
            else throw new Exception("multiple streams present, please specify Pid or Stream");
            foreach (var item in hdr_m.play_item)
                foreach (var audio in item.audio)
                {
                    if (audio.coding_type != 0x80 /* LPCM */) continue;
                    if (audio.pid == chosenPid)
                    {
                        var parent = Directory.GetParent(System.IO.Path.GetDirectoryName(_path));
                        var m2ts = System.IO.Path.Combine(
                            System.IO.Path.Combine(parent.FullName, "STREAM"), 
                            item.clip_id + ".m2ts");
                        var entry = new BDLPCMReader(m2ts, null, chosenPid);
                        readers.Add(entry);
                        break;
                    }
                }
            currentReader = readers[0];
            pcm = currentReader.PCM;
        }

        MPLSHeader parseHeader(FrameReader fr)
        {
            var hdr = new MPLSHeader();
            hdr.play_item = new List<MPLSPlaylistItem>();
            hdr.play_mark = new List<MPLSPlaylistMark>();
            long length = fr.Length;
            hdr.type_indicator = fr.read_uint();
            hdr.type_indicator2 = fr.read_uint();
            hdr.list_pos = fr.read_uint();
            hdr.mark_pos = fr.read_uint();
            hdr.ext_pos = fr.read_uint();
            if (hdr.type_indicator != 0x4d504c53 /*MPLS*/) throw new NotSupportedException();
            if (hdr.type_indicator2 != 0x30313030 && hdr.type_indicator2 != 0x30323030) throw new NotSupportedException();
            if (hdr.list_pos > length || hdr.mark_pos > length || hdr.ext_pos > length) throw new NotSupportedException();
            return hdr;
        }

        void parsePlaylist(FrameReader parentFr)
        {
            uint len = parentFr.read_uint();
            FrameReader fr = new FrameReader(parentFr, len);
            parentFr.skip(len);
            ushort reserved = fr.read_ushort();
            hdr_m.list_count = fr.read_ushort();
            hdr_m.sub_count = fr.read_ushort();
            for (int i = 0; i < hdr_m.list_count; i++)
                hdr_m.play_item.Add(parsePlaylistItem(fr));
        }

        void parsePlaylistMarks(FrameReader parentFr)
        {
            uint len = parentFr.read_uint();
            FrameReader fr = new FrameReader(parentFr, len);
            parentFr.skip(len);
            hdr_m.mark_count = fr.read_ushort();
            for (int ii = 0; ii < hdr_m.mark_count; ii++)
                hdr_m.play_mark.Add(parsePlaylistMark(fr));
        }

        MPLSPlaylistItem parsePlaylistItem(FrameReader parentFr)
        {
            var item = new MPLSPlaylistItem();
            item.video = new List<MPLSStream>();
            item.audio = new List<MPLSStream>();
            item.pg = new List<MPLSStream>();

            // PlayItem Length
            ushort len = parentFr.read_ushort();
            FrameReader fr = new FrameReader(parentFr, len);
            parentFr.skip(len);

            // Primary Clip identifer
            var clip_id = fr.read_bytes(5);
            item.clip_id = Encoding.UTF8.GetString(clip_id, 0, clip_id.Length);

            // skip the redundant "M2TS" CodecIdentifier
            uint codecId = fr.read_uint();
            if (codecId != 0x4D325453) throw new NotSupportedException("Incorrect CodecIdentifier");

            ushort flags = fr.read_ushort();
            bool is_multi_angle = ((flags >> 4) & 1) != 0;
            item.connection_condition = (byte)(flags & 15);
            if (item.connection_condition != 0x01 &&
                item.connection_condition != 0x05 &&
                item.connection_condition != 0x06)
                throw new NotSupportedException("Unexpected connection condition");

            item.stc_id = fr.read_byte();
            item.in_time = fr.read_uint();
            item.out_time = fr.read_uint();

            // Skip UO_mask_table, random_access_flag, reserved, still_mode
            // and still_time
            fr.skip(12);

            if (is_multi_angle)
            {
                byte num_angles = fr.read_byte();
                // skip reserved, is_different_audio, is_seamless_angle_change
                fr.skip(1);
                for (int ii = 1; ii < num_angles; ii++)
                {
                    // Drop clip_id, clip_codec_id, stc_id
                    fr.skip(10);
                }
            }

            // Skip STN len
            fr.skip(2);

            // Skip 2 reserved bytes
            fr.skip(2);

            item.num_video = fr.read_byte();
            item.num_audio = fr.read_byte();
            item.num_pg = fr.read_byte();
            item.num_ig = fr.read_byte();
            item.num_secondary_audio = fr.read_byte();
            item.num_secondary_video = fr.read_byte();
            item.num_pip_pg = fr.read_byte();

            // 5 reserve bytes
            fr.skip(5);

            for (int ii = 0; ii < item.num_video; ii++)
                item.video.Add(parseStream(fr));
            for (int ii = 0; ii < item.num_audio; ii++)
                item.audio.Add(parseStream(fr));
            for ( int ii = 0; ii < item.num_pg; ii++)
                item.pg.Add(parseStream(fr));

            return item;
        }

        MPLSStream parseStream(FrameReader parentFr)
        {
            MPLSStream s = new MPLSStream();
            
            byte len = parentFr.read_byte();
            FrameReader fr = new FrameReader(parentFr, len);
            parentFr.skip(len);

            s.stream_type = fr.read_byte();
            switch (s.stream_type)
            {
                case 1:
                    s.pid = fr.read_ushort();
                    break;
                case 2:
                case 4:
                    s.subpath_id = fr.read_byte();
                    s.subclip_id = fr.read_byte();
                    s.pid = fr.read_ushort();
                    break;
                case 3:
                    s.subpath_id = fr.read_byte();
                    s.pid = fr.read_ushort();
                    break;
                default:
                    throw new Exception("unrecognized stream type");
            };

            len = parentFr.read_byte();
            fr = new FrameReader(parentFr, len);
            parentFr.skip(len);

            s.coding_type = fr.read_byte();
            //value_map_t codec_map[] = {
            //    {0x01, "MPEG-1 Video"},
            //    {0x02, "MPEG-2 Video"},
            //    {0x03, "MPEG-1 Audio"},
            //    {0x04, "MPEG-2 Audio"},
            //    {0x80, "LPCM"},
            //    {0x81, "AC-3"},
            //    {0x82, "DTS"},
            //    {0x83, "TrueHD"},
            //    {0x84, "AC-3 Plus"},
            //    {0x85, "DTS-HD"},
            //    {0x86, "DTS-HD Master"},
            //    {0xea, "VC-1"},
            //    {0x1b, "H.264"},
            //    {0x90, "Presentation Graphics"},
            //    {0x91, "Interactive Graphics"},
            //    {0x92, "Text Subtitle"},
            //    {0, NULL}
            //};
            if (s.coding_type == 0x01
                || s.coding_type == 0x02
                || s.coding_type == 0xea
                || s.coding_type == 0x1b)
            {
                byte fmt = fr.read_byte();
                s.format = (byte)(fmt >> 4);
                s.rate = (byte)(fmt & 15);
            }
            else if (s.coding_type == 0x03
                || s.coding_type == 0x04
                || s.coding_type == 0x80
                || s.coding_type == 0x81
                || s.coding_type == 0x82
                || s.coding_type == 0x83
                || s.coding_type == 0x84
                || s.coding_type == 0x85
                || s.coding_type == 0x86)
            {
                // Audio
                //value_map_t audio_format_map[] = {
                //    {0, "Reserved1"},
                //    {1, "Mono"},
                //    {2, "Reserved2"},
                //    {3, "Stereo"},
                //    {4, "Reserved3"},
                //    {5, "Reserved4"},
                //    {6, "Multi Channel"},
                //    {12, "Combo"},
                //    {0, NULL}
                //};
                //value_map_t audio_rate_map[] = {
                //    {0, "Reserved1"},
                //    {1, "48 Khz"},
                //    {2, "Reserved2"},
                //    {3, "Reserved3"},
                //    {4, "96 Khz"},
                //    {5, "192 Khz"},
                //    {12, "48/192 Khz"},
                //    {14, "48/96 Khz"},
                //    {0, NULL}
                //};
                byte fmt = fr.read_byte();
                s.format = (byte)(fmt >> 4);
                s.rate = (byte)(fmt & 15);
                s.lang = fr.read_bytes(3);
            }
            else if (s.coding_type == 0x90
                || s.coding_type == 0x91)
            {
                s.lang = fr.read_bytes(3);
            }
            else if (s.coding_type == 0x92)
            {
                s.char_code = fr.read_byte();
                s.lang = fr.read_bytes(3);
            }
            else throw new Exception("unrecognized stream type");

            return s;
        }

        MPLSPlaylistMark parsePlaylistMark(FrameReader fr)
        {
            var mark = new MPLSPlaylistMark();
            mark.mark_id = fr.read_byte();
            mark.mark_type = fr.read_byte();
            mark.play_item_ref = fr.read_ushort();
            mark.time = fr.read_uint();
            mark.entry_es_pid = fr.read_ushort();
            mark.duration = fr.read_uint();
            return mark;
        }

        public AudioDecoderSettings Settings { get { return settings; } }

        public void Close()
        {
            foreach (var rdr in readers)
            {
                rdr.Close();
            }
            readers = null;
            currentReader = null;
            _IO = null;
        }

        public long Length
        {
            get
            {
                return -1;
            }
        }

        public long Remaining
        {
            get
            {
                return -1;
            }
        }

        public long Position
        {
            get
            {
                long res = 0;
                foreach (var rdr in readers)
                {
                    res += rdr.Position;
                    if (rdr == currentReader) break;
                }
                return res;
            }
            set
            {
                throw new NotSupportedException();
            }
        }

        public unsafe AudioPCMConfig PCM
        {
            get {
                if (readers == null) openEntries();
                return pcm;
            }
        }

        public string Path { get { return _path; } }

        public unsafe int Read(AudioBuffer buff, int maxLength)
        {
            if (readers == null) openEntries();
            int res = currentReader.Read(buff, maxLength);
            if (res == 0)
            {
                bool nextOne = false;
                foreach (var rdr in readers)
                {
                    if (nextOne)
                    {
                        currentReader = rdr;
                        return currentReader.Read(buff, maxLength);
                    }
                    nextOne = (rdr == currentReader);
                }
                currentReader = null;
            }
            return res;
        }

        string _path;
        Stream _IO;
        byte[] contents;

        AudioPCMConfig pcm;
        List<BDLPCMReader> readers;
        BDLPCMReader currentReader;
        MPLSHeader hdr_m;
        BDLPCMReaderSettings settings;
    }

    internal struct MPLSPlaylistMark
    {
        public byte mark_id;
        public byte mark_type;
        public ushort play_item_ref;
        public uint time;
        public ushort entry_es_pid;
        public uint duration;
    }

    internal struct MPLSStream
    {
        public byte stream_type;
        public byte coding_type;
        public ushort pid;
        public byte subpath_id;
        public byte subclip_id;
        public byte format;
        public byte rate;
        public byte char_code;
        public byte[] lang;
    }

    internal struct MPLSPlaylistItem
    {
        public string clip_id;
        public byte connection_condition;
        public byte stc_id;
        public uint in_time;
        public uint out_time;

        public byte num_video;
        public byte num_audio;
        public byte num_pg;
        public byte num_ig;
        public byte num_secondary_audio;
        public byte num_secondary_video;
        public byte num_pip_pg;
        public List<MPLSStream> video;
        public List<MPLSStream> audio;
        public List<MPLSStream> pg;
    }

    internal struct MPLSHeader
    {
        public uint type_indicator;
        public uint type_indicator2;
        public uint list_pos;
        public uint mark_pos;
        public uint ext_pos;

        public ushort list_count;
        public ushort sub_count;
        public ushort mark_count;

        public List<MPLSPlaylistItem> play_item;
        public List<MPLSPlaylistMark> play_mark;
    }
}
