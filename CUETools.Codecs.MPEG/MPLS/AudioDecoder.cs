using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CUETools.Codecs.MPEG.MPLS
{
    public class AudioDecoder : IAudioSource, IAudioTitleSet
    {
        public unsafe AudioDecoder(DecoderSettings settings, string path, Stream IO)
        {
            m_settings = settings;
            _path = path;
            _IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000);
            int length = (int)_IO.Length;
            if (length > 0x100000) throw new Exception("File too big");
            contents = new byte[length];
            if (_IO.Read(contents, 0, length) != length) throw new Exception("");
            fixed (byte* ptr = &contents[0])
            {
                var fr = new FrameReader(ptr, length);
                hdr_m = parseHeader(fr);
                fr = new FrameReader(ptr + hdr_m.list_pos, length - hdr_m.list_pos);
                parsePlaylist(fr);
                fr = new FrameReader(ptr + hdr_m.mark_pos, length - hdr_m.mark_pos);
                parsePlaylistMarks(fr);
            }
        }

        void openEntries()
        {
            readers = new List<IAudioSource>();
            var pids = new List<int>();
            foreach (var item in hdr_m.play_item)
                foreach (var audio in item.audio)
                {
                    if (audio.coding_type != 0x80 /* LPCM */) continue;
                    pids.Add(audio.pid);
                }
            int chosenPid;
            if (m_settings.StreamId.HasValue)
            {
                if (!pids.Contains(m_settings.StreamId.Value))
                    throw new Exception("StreamId can be " +
                        string.Join(", ", pids.ConvertAll(pid => pid.ToString()).ToArray()));
                chosenPid = m_settings.StreamId.Value;
            }
            else if (m_settings.Stream.HasValue)
            {
                if (m_settings.Stream.Value < 0 || m_settings.Stream.Value >= pids.Count)
                    throw new Exception("Stream can be 0.." + (pids.Count - 1).ToString());
                chosenPid = pids[m_settings.Stream.Value];
            }
            else throw new Exception("multiple streams present, please specify StreamId or Stream");
            foreach (var item in hdr_m.play_item)
                foreach (var audio in item.audio)
                {
                    if (audio.coding_type != 0x80 /* LPCM */) continue;
                    if (m_settings.IgnoreShortItems && item.out_time - item.in_time < shortItemDuration) continue;
                    if (audio.pid == chosenPid)
                    {
                        var parent = Directory.GetParent(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_path)));
                        var m2ts = System.IO.Path.Combine(
                            System.IO.Path.Combine(parent.FullName, "STREAM"), 
                            item.clip_id + ".m2ts");
                        var settings = new BDLPCM.DecoderSettings() { StreamId = chosenPid };
                        var entry = settings.Open(m2ts);
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
            var fr = new FrameReader(parentFr, len);
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
            var fr = new FrameReader(parentFr, len);
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
            var fr = new FrameReader(parentFr, len);
            parentFr.skip(len);

            // Primary Clip identifer
            item.clip_id = fr.read_string(5);

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
            var fr = new FrameReader(parentFr, len);
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
            if (s.coding_type == 0x01
                || s.coding_type == 0x02
                || s.coding_type == 0xea
                || s.coding_type == 0x1b)
            {
                // Video
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
                byte fmt = fr.read_byte();
                s.format = (byte)(fmt >> 4);
                s.rate = (byte)(fmt & 15);
                s.lang = fr.read_string(3);
            }
            else if (s.coding_type == 0x90
                || s.coding_type == 0x91)
            {
                s.lang = fr.read_string(3);
            }
            else if (s.coding_type == 0x92)
            {
                s.char_code = fr.read_byte();
                s.lang = fr.read_string(3);
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

        public IAudioDecoderSettings Settings => m_settings;

        public void Close()
        {
            if (readers != null)
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

        public TimeSpan Duration
        {
            get
            {
                uint totalLength = 0;
                foreach (var item in hdr_m.play_item)
                {
                    if (item.num_audio == 0) continue;
                    uint item_duration = item.out_time - item.in_time;
                    if (m_settings.IgnoreShortItems && item_duration < shortItemDuration) continue;
                    totalLength += item_duration;
                }

                return TimeSpan.FromSeconds(totalLength / 45000.0);
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

        public string FileName
        {
            get
            {
                return System.IO.Path.GetFileName(_path);
            }
        }

        public MPLSHeader MPLSHeader
        {
            get
            {
                return hdr_m;
            }
        }

        public List<IAudioTitle> AudioTitles
        {
            get
            {
                var titles = new List<IAudioTitle>();
                foreach (var item in hdr_m.play_item)
                    foreach (var audio in item.audio)
                    {
                        //if (audio.coding_type != 0x80 /* LPCM */) continue;
                        titles.Add(new AudioTitle(this, audio.pid));
                    }
                return titles;
            }
        }

        public List<TimeSpan> Chapters
        {
            get
            {
                //settings.IgnoreShortItems
                var res = new List<TimeSpan>();
                if (hdr_m.play_mark.Count < 1) return res;
                if (hdr_m.play_item.Count < 1) return res;
                res.Add(TimeSpan.Zero);
                for (int i = 0; i < hdr_m.mark_count; i++)
                {
                    ushort mark_item = hdr_m.play_mark[i].play_item_ref;
                    uint item_in_time = hdr_m.play_item[mark_item].in_time;
                    uint item_out_time = hdr_m.play_item[mark_item].out_time;
                    if (m_settings.IgnoreShortItems && item_out_time - item_in_time < shortItemDuration) continue;
                    uint item_offset = 0;
                    for (int j = 0; j < mark_item; j++)
                    {
                        if (hdr_m.play_item[j].num_audio == 0) continue;
                        uint item_duration = hdr_m.play_item[j].out_time - hdr_m.play_item[j].in_time;
                        if (m_settings.IgnoreShortItems && item_duration < shortItemDuration) continue;
                        item_offset += item_duration;
                    }
                    res.Add(TimeSpan.FromSeconds((hdr_m.play_mark[i].time - item_in_time + item_offset) / 45000.0));
                }
                uint end_offset = 0;
                for (int j = 0; j < hdr_m.play_item.Count; j++)
                {
                    if (hdr_m.play_item[j].num_audio == 0) continue;
                    uint item_duration = hdr_m.play_item[j].out_time - hdr_m.play_item[j].in_time;
                    if (m_settings.IgnoreShortItems && item_duration < shortItemDuration) continue;
                    end_offset += hdr_m.play_item[j].out_time - hdr_m.play_item[j].in_time;
                }
                res.Add(TimeSpan.FromSeconds(end_offset / 45000.0));
                while (res.Count > 1 && res[1] - res[0] < TimeSpan.FromSeconds(1.0)) res.RemoveAt(1);
                while (res.Count > 1 && res[res.Count - 1] - res[res.Count - 2] < TimeSpan.FromSeconds(1.0)) res.RemoveAt(res.Count - 2);
                return res;
            }
        }

        readonly static int shortItemDuration = 45000 * 30;

        string _path;
        Stream _IO;
        byte[] contents;

        AudioPCMConfig pcm;
        List<IAudioSource> readers;
        IAudioSource currentReader;
        MPLSHeader hdr_m;
        DecoderSettings m_settings;
    }

    public class AudioTitle : IAudioTitle
    {
        public AudioTitle(AudioDecoder source, int pid)
        {
            this.source = source;
            this.pid = pid;
        }

        public List<TimeSpan> Chapters => source.Chapters;
        public AudioPCMConfig PCM
        {
            get
            {
                var s = FirstStream;
                int channelCount = s.format == 1 ? 1 : s.format == 3 ? 2 : s.format == 6 ? 5 : 0;
                int sampleRate = s.rate == 1 ? 48000 : s.rate == 4 ? 96000 : s.rate == 5 ? 192000 : s.rate == 12 ? 192000 : s.rate == 14 ? 96000 : 0;
                int bitsPerSample = 0;
                return new AudioPCMConfig(bitsPerSample, channelCount, sampleRate);
            }
        }
        public string Codec
        {
            get
            {
                var s = FirstStream;
                return s != null ? s.CodecString : "?";
            }
        }
        public string Language
        {
            get
            {
                var s = FirstStream;
                return s != null ? s.LanguageString : "?";
            }
        }
        public int StreamId => pid;

        MPLSStream FirstStream
        {
            get
            {
                MPLSStream result = null;
                source.MPLSHeader.play_item.ForEach(i => i.audio.FindAll(a => a.pid == pid).ForEach(x => result = x));
                return result;
            }
        }

        AudioDecoder source;
        int pid;
    }

    public struct MPLSPlaylistMark
    {
        public byte mark_id;
        public byte mark_type;
        public ushort play_item_ref;
        public uint time;
        public ushort entry_es_pid;
        public uint duration;
    }

    public class MPLSStream
    {
        public byte stream_type;
        public byte coding_type;
        public ushort pid;
        public byte subpath_id;
        public byte subclip_id;
        public byte format;
        public byte rate;
        public byte char_code;
        public string lang;

        public string FormatString
        {
            get
            {
                if (coding_type == 0x01
                    || coding_type == 0x02
                    || coding_type == 0xea
                    || coding_type == 0x1b)
                    switch (format)
                    {
                        case 0: return "reserved0";
                        case 1: return "480i";
                        case 2: return "576i";
                        case 3: return "480p";
                        case 4: return "1080i";
                        case 5: return "720p";
                        case 6: return "1080p";
                        case 7: return "576p";
                        default: return format.ToString();
                    }
                switch (format)
                {
                    case 0: return "reserved0";
                    case 1: return "mono";
                    case 2: return "reserved2";
                    case 3: return "stereo";
                    case 4: return "reserved4";
                    case 5: return "reserved5";
                    case 6: return "multi-channel";
                    case 12: return "combo";
                    default: return format.ToString();
                }
            }
        }

        public int FrameRate
        {
            get
            {
                switch (rate)
                {
                    case 1: return 24;
                    case 2: return 24;
                    case 3: return 25;
                    case 4: return 30;
                    case 6: return 50;
                    case 7: return 60;
                    default: throw new NotSupportedException();
                }
            }
        }

        public bool Interlaced
        {
            get
            {
                return format == 1 || format == 2 || format == 4;
            }
        }

        public string RateString
        {
            get
            {
                if (coding_type == 0x01
                    || coding_type == 0x02
                    || coding_type == 0xea
                    || coding_type == 0x1b)
                    switch (rate)
                    {
                        case 0: return "reserved0";
                        case 1: return "23.976";
                        case 2: return "24";
                        case 3: return "25";
                        case 4: return "29.97";
                        case 5: return "reserved5";
                        case 6: return "50";
                        case 7: return "59.94";
                        default: return rate.ToString();
                    }
                switch (rate)
                {
                    case 0: return "reserved0";
                    case 1: return "48KHz";
                    case 2: return "reserved2";
                    case 3: return "reserved3";
                    case 4: return "96KHz";
                    case 5: return "192KHz";
                    //case 12: return "48/192KHz"; (core/hd)
                    case 12: return "192KHz";
                    //case 14: return "48/96KHz"; (core/hd)
                    case 14: return "96KHz";
                    default: return rate.ToString();
                }
            }
        }

        public string CodecString
        {
            get
            {
                switch (coding_type)
                {
                    case 0x01: return "MPEG-1 Video";
                    case 0x02: return "MPEG-2 Video";
                    case 0x03: return "MPEG-1 Audio";
                    case 0x04: return "MPEG-2 Audio";
                    //case 0x80: return "LPCM";
                    case 0x80: return "RAW/PCM";
                    case 0x81: return "AC-3";
                    case 0x82: return "DTS";
                    case 0x83: return "TrueHD";
                    case 0x84: return "AC-3 Plus";
                    case 0x85: return "DTS-HD";
                    //case 0x86: return "DTS-HD Master";
                    case 0x86: return "DTS Master Audio";
                    case 0xea: return "VC-1";
                    case 0x1b: return "h264/AVC";
                    case 0x90: return "Presentation Graphics";
                    case 0x91: return "Interactive Graphics";
                    case 0x92: return "Text Subtitle";
                    default: return coding_type.ToString();
                }
            }
        }

        public byte CodingType
        {
            get
            {
                return coding_type;
            }
        }

        public byte FormatType
        {
            get
            {
                return format;
            }
        }

        public string LanguageString
        {
            get
            {
                CultureInfo[] cultures = CultureInfo.GetCultures(CultureTypes.AllCultures);
                foreach (var culture in cultures)
                {
                    // Exclude custom cultures.
                    if ((culture.CultureTypes & CultureTypes.UserCustomCulture) == CultureTypes.UserCustomCulture)
                        continue;

                    if (culture.ThreeLetterISOLanguageName == lang)
                        return culture.EnglishName;
                }
                return lang;
            }
        }
    }

    public struct MPLSPlaylistItem
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

    public struct MPLSHeader
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
