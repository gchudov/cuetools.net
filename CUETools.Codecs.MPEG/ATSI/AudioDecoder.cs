using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using static CUETools.Codecs.AudioPCMConfig;

namespace CUETools.Codecs.MPEG.ATSI
{
    public class AudioDecoder : IAudioSource, IAudioContainer
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
                fr = new FrameReader(ptr, length);
                parseTitles(fr);
            }
        }

        void openEntries()
        {
            readers = new List<IAudioSource>();
            var pids = new List<int>();
            //foreach (var item in hdr_m.play_item)
            //    foreach (var audio in item.audio)
            //    {
            //        if (audio.coding_type != 0x80 /* LPCM */) continue;
            //        pids.Add(audio.pid);
            //    }
            //int chosenPid;
            //if (m_settings.StreamId.HasValue)
            //{
            //    if (!pids.Contains(m_settings.StreamId.Value))
            //        throw new Exception("StreamId can be " +
            //            string.Join(", ", pids.ConvertAll(pid => pid.ToString()).ToArray()));
            //    chosenPid = m_settings.StreamId.Value;
            //}
            //else if (m_settings.Stream.HasValue)
            //{
            //    if (m_settings.Stream.Value < 0 || m_settings.Stream.Value >= pids.Count)
            //        throw new Exception("Stream can be 0.." + (pids.Count - 1).ToString());
            //    chosenPid = pids[m_settings.Stream.Value];
            //}
            //else throw new Exception("multiple streams present, please specify StreamId or Stream");
            //foreach (var item in hdr_m.play_item)
            //    foreach (var audio in item.audio)
            //    {
            //        if (audio.coding_type != 0x80 /* LPCM */) continue;
            //        if (m_settings.IgnoreShortItems && item.out_time - item.in_time < shortItemDuration) continue;
            //        if (audio.pid == chosenPid)
            //        {
            //            var parent = Directory.GetParent(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_path)));
            //            var m2ts = System.IO.Path.Combine(
            //                System.IO.Path.Combine(parent.FullName, "STREAM"), 
            //                item.clip_id + ".m2ts");
            //            var settings = new BDLPCM.DecoderSettings() { StreamId = chosenPid };
            //            var entry = settings.Open(m2ts);
            //            readers.Add(entry);
            //            break;
            //        }
            //    }
            currentReader = readers[0];
            pcm = currentReader.PCM;
        }

        ATSIHeader parseHeader(FrameReader fr)
        {
            var hdr = new ATSIHeader(fr);

            uint aob_offset = 0;
            for (int i = 0; i < 9; i++)
            {
                var aob = new DVDAAOBFile();
                aob.fileName = System.IO.Path.Combine(
                    System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(_path)),
                    $"ATS_{System.IO.Path.GetFileNameWithoutExtension(_path).Substring(4, 2)}_{i + 1}.AOB");
                aob.first = aob_offset;
                try
                {
                    aob.atsFile = new FileStream(aob.fileName, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000);
                    aob.last = (uint)(aob.first + aob.atsFile.Length / DVDA.BLOCK_SIZE) - 1;
                    aob.isExist = true;
                }
                catch (FileNotFoundException)
                {
                    aob.last = aob.first + (uint)((1024 * 1024 - 32) * 1024 / DVDA.BLOCK_SIZE - 1);
                    aob.isExist = false;
                }
                aob_offset = aob.last + 1;
                hdr.aobs.Add(aob);
            }

            return hdr;
        }

        unsafe void parseTitles(FrameReader parentFr)
        {
            for (int i = 0; i < hdr_m.nr_of_titles; i++)
            {
                var fr = new FrameReader(parentFr.Ptr + 0x800 + hdr_m.ats_title_idx[i], parentFr.Length - 0x800 - hdr_m.ats_title_idx[i]);
                hdr_m.titles.Add(parseTitle(fr));
            }
            for (int i = 0; i < hdr_m.nr_of_titles; i++)
            {
                var fr = new FrameReader(parentFr.Ptr + 0x100 + i * 16, 16);
                hdr_m.titles[i].codec = fr.read_ushort();
                hdr_m.titles[i].format = fr.read_uint();
            }
        }

        ATSITitle parseTitle(FrameReader fr)
        {
            var titleFr = new FrameReader(fr, fr.Length);
            var title = new ATSITitle(this, titleFr);
            for (int i = 0; i < title.ntracks; i++)
                title.track_timestamp.Add(new ATSITrackTimestamp(titleFr));
            fr.skip(title.track_sector_table_offset);
            for (int i = 0; i < title.nindexes; i++)
            {
                var dvdaSectorPointer = new ATSITrackSector(fr);
                title.track_sector.Add(dvdaSectorPointer);
                for (int k = 0; k < title.ntracks; k++)
                {
                    var track_curr_idx = title.track_timestamp[k].pg_id;
                    var track_next_idx = (k < title.ntracks - 1) ? title.track_timestamp[k + 1].pg_id : 0;
                    if (i + 1 >= track_curr_idx && (i + 1 < track_next_idx || track_next_idx == 0))
                    {
                        title.track_timestamp[k].sector_pointers.Add(i);
                    }
                }

                var nblocks = Math.Min(SEGMENT_HEADER_BLOCKS, dvdaSectorPointer.last_sector - dvdaSectorPointer.first_sector + 1);
                var head_buf = new byte[nblocks * DVDA.BLOCK_SIZE];
                for (int b = 0; b < nblocks; b++)
                {
                    getBlock(dvdaSectorPointer.first_sector + b, head_buf, b * DVDA.BLOCK_SIZE);
                }
                dvdaSectorPointer.dvdaBlock = new DVDABlock(head_buf);
            }
            return title;
        }

        void getBlock(long block_no, byte[] buf, int offset)
        {
            var aob = hdr_m.aobs.Find(a => a.isExist && block_no >= a.first && block_no <= a.last);
            if (aob == null) return;
            aob.atsFile.Seek((block_no - aob.first) * DVDA.BLOCK_SIZE, SeekOrigin.Begin);
            if (aob.atsFile.Read(buf, offset, DVDA.BLOCK_SIZE) != DVDA.BLOCK_SIZE)
                throw new Exception();
            // theZone->decryptBlock(buf_ptr);
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
                int title = 0;
                if (!m_settings.Title.HasValue)
                {
                    if (hdr_m.titles.Count > 1) throw new Exception("multiple titles present, please specify Title");
                }
                else
                {
                    if (m_settings.Title.Value < 0 || m_settings.Title >= hdr_m.titles.Count)
                        throw new Exception($"Title can be 0..{hdr_m.titles.Count - 1}");
                    title = m_settings.Title.Value;
                }
                var chapters = hdr_m.titles[title].Chapters;
                return chapters[chapters.Count - 1];
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

        public List<IAudioTitle> AudioTitles => hdr_m.titles.ConvertAll(x => x as IAudioTitle);

        public ATSIHeader ATSIHeader
        {
            get
            {
                return hdr_m;
            }
        }

        readonly static int SEGMENT_HEADER_BLOCKS = 16;

        string _path;
        Stream _IO;
        byte[] contents;

        AudioPCMConfig pcm;
        List<IAudioSource> readers;
        IAudioSource currentReader;
        ATSIHeader hdr_m;
        DecoderSettings m_settings;
    }

    public class ATSITitle : IAudioTitle
    {
        internal ATSITitle(AudioDecoder atsi, FrameReader fr)
        {
            this.atsi = atsi;

            track_timestamp = new List<ATSITrackTimestamp>();
            track_sector = new List<ATSITrackSector>();

            unknown1 = fr.read_ushort();
            ntracks = fr.read_byte();
            nindexes = fr.read_byte();
            length_pts = fr.read_uint();
            unknown2 = fr.read_ushort();
            unknown3 = fr.read_ushort();
            track_sector_table_offset = fr.read_ushort();
            unknown4 = fr.read_ushort();
        }

        // ?Unknown (e.g. 0x0000)
        public ushort unknown1;
        // ???Number of tracks in title (repeated - e.g. 0x0303 for 3 tracks, 0x0b0b for 12 tracks)
        public byte ntracks;
        public byte nindexes;
        // Length of track in PTS ticks
        public uint length_pts;
        // ?Unknown (e.g. 0x0000)
        public ushort unknown2;
        // ?Unknown (e.g. 0x0010)
        public ushort unknown3;
        // Byte pointer to start of sector pointers table (relative to the start of this title record)
        public ushort track_sector_table_offset;
        // ?Unknown (e.g. 0x0000)
        public ushort unknown4;

        public const int SIZE = 16;

        AudioDecoder atsi;

        public ushort codec;
        public uint format;

        public byte StreamId
        {
            get
            {
                if (track_sector.Count < 1) return 0;
                if (track_sector[0].dvdaBlock == null) return 0;
                if (track_sector[0].dvdaBlock.sh == null) return 0;
                return track_sector[0].dvdaBlock.sh.stream_id;
            }
        }

        public string CodecString
        {
            get
            {
                switch (StreamId)
                {
                    case DVDA.PCM_STREAM_ID: return "RAW/PCM";
                    case DVDA.MLP_STREAM_ID: return "MLP";
                    default: return StreamId.ToString();
                }
                //switch (codec)
                //{
                //    case 0x000: return "RAW/PCM";
                //    case 0x100: return "MLP";
                //    default: return codec.ToString();
                //}
            }
        }

        public AudioPCMConfig PCM
        {
            get
            {
                if (track_sector.Count < 1) return null;
                if (track_sector[0].dvdaBlock == null) return null;
                return new AudioPCMConfig(
                    track_sector[0].dvdaBlock.gr1_bits,
                    track_sector[0].dvdaBlock.channels,
                    track_sector[0].dvdaBlock.gr1_frequency);
            }
        }

        public string RateString
        {
            get
            {
                var sr = PCM.SampleRate;
                if (sr % 1000 == 0) return $"{sr / 1000}KHz";
                if (sr % 100 == 0) return $"{sr / 100}.{(sr / 100) % 10}KHz";
                return $"{sr}Hz";
            }
        }

        public string FormatString
        {
            get
            {
                if (track_sector.Count < 1) return "?";
                if (track_sector[0].dvdaBlock == null) return "?";
                switch (track_sector[0].dvdaBlock.ch_assignment)
                {
                    case 0: return "mono";
                    case 1: return "stereo";
                    default: return "multi-channel";
                }
            }
        }

        public List<TimeSpan> Chapters
        {
            get
            {
                //settings.IgnoreShortItems
                var res = new List<TimeSpan>();
                double time_base = 90000.0;
                var durations = track_timestamp.ConvertAll(track => ((long)track.len_in_pts));
                bool ignoreShortItems = true; // m_settings.IgnoreShortItems 
                const int shortItemDuration = 90000 * 10;
                for (int i = 0; i < track_timestamp.Count; i++)
                {
                    if (ignoreShortItems && durations[i] < shortItemDuration) continue;
                    long item_offset = 0;
                    for (int j = 0; j < i; j++)
                    {
                        var item_duration = durations[j];
                        if (ignoreShortItems && item_duration < shortItemDuration) continue;
                        item_offset += item_duration;
                    }
                    res.Add(TimeSpan.FromSeconds((uint)item_offset / time_base));
                }
                long end_offset = 0;
                for (int j = 0; j < track_timestamp.Count; j++)
                {
                    var item_duration = durations[j];
                    if (ignoreShortItems && item_duration < shortItemDuration) continue;
                    end_offset += item_duration;
                }
                res.Add(TimeSpan.FromSeconds((uint)end_offset / time_base));
                //while (res.Count > 1 && res[1] - res[0] < 45000) res.RemoveAt(1);
                //while (res.Count > 1 && res[res.Count - 1] - res[res.Count - 2] < 45000) res.RemoveAt(res.Count - 2);
                return res;
            }
        }

        public List<ATSITrackTimestamp> track_timestamp;
        public List<ATSITrackSector> track_sector;
    }

    public class ATSITrackTimestamp
    {
        internal ATSITrackTimestamp(FrameReader fr)
        {
            sector_pointers = new List<int>();
            track_nr = fr.read_ushort();
            unknown2 = fr.read_ushort();
            pg_id = fr.read_byte();
            unknown3 = fr.read_byte();
            first_pts = fr.read_uint();
            len_in_pts = fr.read_uint();
            padding1 = fr.read_ushort();
            padding2 = fr.read_uint();
        }

        // ?Unknown(e.g. 0xc010 for first track, and 0x0010 for subsequent)
        public ushort track_nr;
        // ?Unknown(e.g. 0x0000)
        public ushort unknown2;
        // Track number in title
        public byte pg_id;
        // ?Unknown(e.g. 0x00)
        public byte unknown3;
        // First PTS of track
        public uint first_pts;
        //Length of track in PTS ticks
        public uint len_in_pts;
        //Padding(zero)
        public ushort padding1;
        //Padding(zero)
        public uint padding2;

        public const int SIZE = 20;

        public List<int> sector_pointers;
    }

    public class DVDABlock
    {
        public DVDABlock(byte[] p_blocks)
        {
            head_buf = new byte[p_blocks.Length];
            head_len = 0;
            scrambled = false;
            int n_blocks = p_blocks.Length / 2048;
            for (int i = 0; i < n_blocks; i++)
                getPS1Payload(p_blocks, i * DVDA.BLOCK_SIZE);
            if (head_len > 0)
                getPS1Params(head_buf, head_len);
        }

        void getPS1Payload(byte[] p_block, int offset, bool ignore_scrambling = true)
        {
            int i_ps1_body;
            int i_curr = offset;
            bool scrambling_checked = false;
            if (p_block[i_curr] != 0x00 || p_block[i_curr + 1] != 0x00 || p_block[i_curr + 2] != 0x01 || p_block[i_curr + 3] != 0xBA)
                return;
            // scrambled = false;
            i_curr += 14 + (p_block[i_curr + 13] & 0x07);
            while (i_curr < offset + DVDA.BLOCK_SIZE)
            {
                int pes_len = (p_block[i_curr + 4] << 8) + p_block[i_curr + 5];
                if (p_block[i_curr + 0] != 0x00 || p_block[i_curr + 1] != 0x00 || p_block[i_curr + 2] != 0x01)
                    break;
                uint pes_sid = p_block[i_curr + 3];
                if (pes_sid == 0xbd)
                { // check for private stream 1
                    if (!scrambling_checked && (p_block[i_curr + 6] & 0x30) != 0)
                        scrambled = true; // check for pes_scrambling_control
                    scrambling_checked = true;
                    int i_ps1_header = i_curr + 9 + p_block[i_curr + 8];
                    int i_ps1_end = i_curr + 6 + pes_len;
                    if (scrambled && !ignore_scrambling && (i_ps1_end - offset > 127))
                        throw new Exception("Block scrambled");
                    i_ps1_body = i_ps1_header + ((head_len > 0) ? getSubstreamHeaderLen(p_block, i_ps1_header, i_ps1_end - i_ps1_header) : 0);
                    int ps1_body_len = i_ps1_end - i_ps1_body;
                    if (ps1_body_len > 0)
                    {
                        Array.Copy(p_block, i_ps1_body, head_buf, head_len, ps1_body_len);
                        head_len += ps1_body_len;
                    }
                }
                i_curr += 6 + pes_len;
            }
        }

        unsafe void getPS1Params(byte[] ps1_header, int ps1_len)
        {
            ch_assignment = -1;
            gr1_frequency = 0;
            gr1_bits = 0;
            gr2_frequency = 0;
            gr2_bits = 0;
            vbr = false;
            peak_bitrate = 0;
            substreams = 0;
            cci = 0;
            if (ps1_len == 0)
                return;
            fixed (byte* p_ps1_header = &ps1_header[0])
            {
                var fr = new FrameReader(p_ps1_header, ps1_len);
                sh = new SUB_HEADER(fr);
                switch (sh.stream_id)
                {
                    case DVDA.PCM_STREAM_ID:
                        {
                            if (sh.extra_len < PCM_EXTRAHEADER.SIZE) break;
                            PCM_EXTRAHEADER pcm_ehdr = new PCM_EXTRAHEADER(fr);
                            cci = pcm_ehdr.cci;
                            ch_assignment = pcm_ehdr.channel_assignment;
                            decode_grp1_bits(pcm_ehdr.group1_bits);
                            decode_grp2_bits(pcm_ehdr.group2_bits);
                            decode_grp1_freq(pcm_ehdr.group1_freq);
                            decode_grp2_freq(pcm_ehdr.group2_freq);
                            vbr = false;
                            peak_bitrate = gr1_channels * gr1_frequency * gr1_bits + gr2_channels * gr2_frequency * gr2_bits;
                            substreams = 1;
                            break;
                        }

                    case DVDA.MLP_STREAM_ID:
                        {
                            if (sh.extra_len < MLP_EXTRAHEADER.SIZE) break;
                            MLP_EXTRAHEADER mlp_ehdr = new MLP_EXTRAHEADER(fr);
                            fr.skip(sh.extra_len - MLP_EXTRAHEADER.SIZE);
                            while (fr.Length > MLP_LINK.SIZE + MLP_SIGNATURE.SIZE)
                            {
                                FrameReader fr1 = new FrameReader(fr, MLP_LINK.SIZE + MLP_SIGNATURE.SIZE);
                                fr1.skip(MLP_LINK.SIZE);
                                MLP_SIGNATURE mlp_sign = new MLP_SIGNATURE(fr1);
                                if (mlp_sign.signature1 != 0xF8726FBB /*|| p_mlp_sign->signature2 != 0xB752*/)
                                {
                                    fr.skip(1);
                                    continue;
                                }
                                cci = mlp_ehdr.cci;
                                fr.skip(MLP_LINK.SIZE);
                                ch_assignment = mlp_sign.channel_assignment;
                                decode_grp1_bits(mlp_sign.group1_bits);
                                decode_grp2_bits(mlp_sign.group2_bits);
                                decode_grp1_freq(mlp_sign.group1_freq);
                                decode_grp2_freq(mlp_sign.group2_freq);
                                vbr = (mlp_sign.bitrate & 0x8000) != 0;
                                peak_bitrate = ((mlp_sign.bitrate & ~0x8000) * gr1_frequency + 8) >> 4;
                                substreams = mlp_sign.substreams;
                                break;
                            }
                            break;
                        }
                }
            }
        }

        unsafe int getSubstreamHeaderLen(byte[] substream_buf, int substream_off, int substream_len)
        {
            if (substream_len <= 4) return 0;
            fixed (byte* p = &substream_buf[substream_off])
            {
                var fr = new FrameReader(p, substream_len);
                var hdr = new SUB_HEADER(fr);
                switch (hdr.stream_id)
                {
                    case DVDA.PCM_STREAM_ID:
                    case DVDA.MLP_STREAM_ID:
                        return SUB_HEADER.SIZE + hdr.extra_len;
                    default:
                        return 0;
                }
            }
        }

        public int gr1_channels => ChannelsInMask(DVDA.grp1_ch_table[ch_assignment]);

        public int gr2_channels => ChannelsInMask(DVDA.grp2_ch_table[ch_assignment]);

        public int channels => gr1_channels + gr2_channels;

        public void decode_grp1_bits(byte b)
        {
            switch (b)
            {
                case 0:
                    gr1_bits = 16;
                    break;
                case 1:
                    gr1_bits = 20;
                    break;
                case 2:
                    gr1_bits = 24;
                    break;
                default:
                    gr1_bits = 0;
                    break;
            }
        }

        public void decode_grp2_bits(byte b)
        {
            switch (b)
            {
                case 0:
                    gr2_bits = 16;
                    break;
                case 1:
                    gr2_bits = 20;
                    break;
                case 2:
                    gr2_bits = 24;
                    break;
                case 0x0f:
                default:
                    gr2_bits = 0;
                    break;
            }
        }

        public void decode_grp1_freq(byte b)
        {
            switch (b)
            {
                case 0:
                    gr1_frequency = 48000;
                    break;
                case 1:
                    gr1_frequency = 96000;
                    break;
                case 2:
                    gr1_frequency = 192000;
                    break;
                case 8:
                    gr1_frequency = 44100;
                    break;
                case 9:
                    gr1_frequency = 88200;
                    break;
                case 0x0A:
                    gr1_frequency = 176400;
                    break;
                default:
                    gr1_frequency = 0;
                    break;
            }
        }

        public void decode_grp2_freq(byte b)
        {
            switch (b)
            {
                case 0:
                    gr2_frequency = 48000;
                    break;
                case 1:
                    gr2_frequency = 96000;
                    break;
                case 8:
                    gr2_frequency = 44100;
                    break;
                case 9:
                    gr2_frequency = 88200;
                    break;
                case 0x0F:
                default:
                    gr2_frequency = 0;
                    break;
            }
        }

        //uint getChannelId(int channel)
        //{
        //    if (channel < gr1_channels + gr2_channels)
        //    {
        //        if (channel < gr1_channels)
        //            return DVDA.grp1_ch_table[ch_assignment].[channel];
        //        else
        //            return DVDA.grp2_ch_table[ch_assignment].[channel - gr1_channels];
        //    }
        //    return 0;
        //}

        //int remapChannel(int channel)
        //{
        //    return ch_remap[ch_assignment][channel];
        //}

        public SUB_HEADER sh;
        public int head_check_ofs;
        public int tail_check_ofs;
        public byte[] head_buf;
        public int head_len;
        public int ch_assignment;
        public int gr1_frequency;
        public int gr1_bits;
        public int gr2_frequency;
        public int gr2_bits;
        public bool vbr;
        public int peak_bitrate;
        public int substreams;
        public byte cci;


        private bool scrambled;
    }

    public class ATSITrackSector
    {
        internal ATSITrackSector(FrameReader fr)
        {
            unknown4 = fr.read_uint();
            first_sector = fr.read_uint();
            last_sector = fr.read_uint();
        }

        // ?? Unknown (e.g. 0x01000000)
        public uint unknown4;
        // Relative sector pointer to first sector of track (relative to the start of the first .AOB file)
        public uint first_sector;
        // Relative sector pointer to last sector of track(relative to the start of the first.AOB file)
        public uint last_sector;

        public DVDABlock dvdaBlock;
    }

    public class SUB_HEADER
    {
        internal SUB_HEADER(FrameReader fr)
        {
            stream_id = fr.read_byte();
            cyclic = fr.read_byte();
            padding1 = fr.read_byte();
            extra_len = fr.read_byte();
        }
        public byte stream_id;
        public byte cyclic;
        public byte padding1;
        public byte extra_len;

        public const int SIZE = 4;
    };

    public struct PCM_EXTRAHEADER
    {
        internal PCM_EXTRAHEADER(FrameReader fr)
        {
            byte tmp;
            first_audio_frame = fr.read_ushort();
            padding1 = fr.read_byte();
            group1_bits = (byte)(((tmp = fr.read_byte()) >> 4) & 0xf);
            group2_bits = (byte)(tmp & 0xf);
            group1_freq = (byte)(((tmp = fr.read_byte()) >> 4) & 0xf);
            group2_freq = (byte)(tmp & 0xf);
            padding2 = fr.read_byte();
            channel_assignment = fr.read_byte();
            padding3 = fr.read_byte();
            cci = fr.read_byte();
        }

        public ushort first_audio_frame;
        public byte padding1;
        public byte group2_bits;// : 4;
        public byte group1_bits;// : 4;
        public byte group2_freq;// : 4;
        public byte group1_freq;// : 4;
        public byte padding2;
        public byte channel_assignment;
        public byte padding3;
        public byte cci;

        public const int SIZE = 9;
    };

    public class MLP_EXTRAHEADER
    {
        internal MLP_EXTRAHEADER(FrameReader fr)
        {
            fr.read_uint();
            cci = fr.read_byte();
        }

        public byte padding1;
        public byte padding2;
        public byte padding3;
        public byte padding4;
        public byte cci;

        public const int SIZE = 5;
    };

    public struct MLP_LINK
    {
        public ushort block_length;// : 12;
        public ushort padding;

        public const int SIZE = 4;
    };

    public struct MLP_SIGNATURE
    {
        internal MLP_SIGNATURE(FrameReader fr)
        {
            byte tmp;
            signature1 = fr.read_uint();
            group1_bits = (byte)(((tmp = fr.read_byte()) >> 4) & 0xf);
            group2_bits = (byte)(tmp & 0xf);
            group1_freq = (byte)(((tmp = fr.read_byte()) >> 4) & 0xf);
            group2_freq = (byte)(tmp & 0xf);
            padding1 = fr.read_byte();
            channel_assignment = fr.read_byte();
            signature2 = fr.read_ushort();
            padding2 = fr.read_uint();
            bitrate = fr.read_ushort();
            substreams = (byte)(fr.read_byte() & 0xf);
        }

        public uint signature1;
        public byte group2_bits;// : 4;
        public byte group1_bits;// : 4;
        public byte group2_freq;// : 4;
        public byte group1_freq;// : 4;
        public byte padding1;
        public byte channel_assignment;
        public ushort signature2;
        public uint padding2;
        public ushort bitrate;
        public byte substreams;// : 4;

        public const int SIZE = 17;
    };

    public class ATSAudioFormat
    {
        public ushort audio_type;
    }

    public class DVDAAOBFile
    {
        public string fileName;
        public Stream atsFile;
        public uint first;
        public uint last;
        public bool isExist;
    }

    // Audio Title Set Information Management Table.
    public class ATSIMAT
    {
        internal ATSIMAT(FrameReader fr)
        {
            ats_audio_format = new ATSAudioFormat[8];
            ats_downmix_matrix = new ushort[16, 8];
            ats_identifier = fr.read_string(12);
            if (ats_identifier != "DVDAUDIO-ATS") throw new NotSupportedException();
            ats_last_sector = fr.read_uint();
            atsi_last_sector = fr.read_uint();
            ats_category = fr.read_uint();
            atsi_last_byte = fr.read_uint();
            atsm_vobs = fr.read_uint();
            atstt_vobs = fr.read_uint();
            ats_ptt_srpt = fr.read_uint();
            ats_pgcit = fr.read_uint();
            atsm_pgci_ut = fr.read_uint();
            ats_tmapt = fr.read_uint();
            atsm_c_adt = fr.read_uint();
            atsm_vobu_admap = fr.read_uint();
            ats_c_adt = fr.read_uint();
            ats_vobu_admap = fr.read_uint();
            for (int i = 0; i < 8; i++)
            {
                ats_audio_format[i] = new ATSAudioFormat();
                ats_audio_format[i].audio_type = fr.read_ushort();
            }
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 8; j++)
                    ats_downmix_matrix[i, j] = fr.read_ushort();
        }

        public string ats_identifier; // [12];
        public uint ats_last_sector;
        public uint atsi_last_sector;
        public uint ats_category;
        public uint atsi_last_byte;
        public uint atsm_vobs;
        public uint atstt_vobs;
        public uint ats_ptt_srpt;
        public uint ats_pgcit;
        public uint atsm_pgci_ut;
        public uint ats_tmapt;
        public uint atsm_c_adt;
        public uint atsm_vobu_admap;
        public uint ats_c_adt;
        public uint ats_vobu_admap;
        public ATSAudioFormat[] ats_audio_format;
        public ushort[,] ats_downmix_matrix;
    }

    public class ATSIHeader
    {
        internal ATSIHeader(FrameReader fr)
        {
            titles = new List<ATSITitle>();
            aobs = new List<DVDAAOBFile>();
            ats_title_idx = new List<uint>();

            var frMat = new FrameReader(fr, fr.Length);
            mat = new ATSIMAT(frMat);

            //if (mat.atsm_vobs == 0)
            //    dvdaTitlesetType = DVDTitlesetAudio;
            //else
            //    dvdaTitlesetType = DVDTitlesetVideo;
            // aobs_last_sector = mat.ats_last_sector - 2 * (mat.atsi_last_sector + 1);

            fr.skip(2048);
            nr_of_titles = fr.read_ushort();
            padding = fr.read_ushort();
            last_byte = fr.read_uint();
            for (int i = 0; i < nr_of_titles; i++)
            {
                // ?? Unknown - e.g. 0x8100 for first title, 0x8200 for second etc etc
                fr.skip(2);
                // ??unknown (e.g. 0x0000 or 0x0100)
                fr.skip(2);
                // Byte offset to record in following table (relative to the start of this sector)
                ats_title_idx.Add(fr.read_uint());
            }
        }

        ATSIMAT mat;

        // audio_pgcit_t
        // Number of titles in the ATS
        public ushort nr_of_titles;
        // Padding (zero)
        public ushort padding;
        // Address of last byte in this table
        public uint last_byte;

        public List<uint> ats_title_idx;
        public List<ATSITitle> titles;
        public List<DVDAAOBFile> aobs;
    }

    public static class DVDA
    {
        public static SpeakerConfig[] grp1_ch_table =
        {
            SpeakerConfig.DVDAUDIO_GR1_0,
            SpeakerConfig.DVDAUDIO_GR1_1,
            SpeakerConfig.DVDAUDIO_GR1_2,
            SpeakerConfig.DVDAUDIO_GR1_3,
            SpeakerConfig.DVDAUDIO_GR1_4,
            SpeakerConfig.DVDAUDIO_GR1_5,
            SpeakerConfig.DVDAUDIO_GR1_6,
            SpeakerConfig.DVDAUDIO_GR1_7,
            SpeakerConfig.DVDAUDIO_GR1_8,
            SpeakerConfig.DVDAUDIO_GR1_9,
            SpeakerConfig.DVDAUDIO_GR1_10,
            SpeakerConfig.DVDAUDIO_GR1_11,
            SpeakerConfig.DVDAUDIO_GR1_12,
            SpeakerConfig.DVDAUDIO_GR1_13,
            SpeakerConfig.DVDAUDIO_GR1_14,
            SpeakerConfig.DVDAUDIO_GR1_15,
            SpeakerConfig.DVDAUDIO_GR1_16,
            SpeakerConfig.DVDAUDIO_GR1_17,
            SpeakerConfig.DVDAUDIO_GR1_18,
            SpeakerConfig.DVDAUDIO_GR1_19,
            SpeakerConfig.DVDAUDIO_GR1_20,
        };

        public static SpeakerConfig[] grp2_ch_table =
        {
            SpeakerConfig.DVDAUDIO_GR2_0,
            SpeakerConfig.DVDAUDIO_GR2_1,
            SpeakerConfig.DVDAUDIO_GR2_2,
            SpeakerConfig.DVDAUDIO_GR2_3,
            SpeakerConfig.DVDAUDIO_GR2_4,
            SpeakerConfig.DVDAUDIO_GR2_5,
            SpeakerConfig.DVDAUDIO_GR2_6,
            SpeakerConfig.DVDAUDIO_GR2_7,
            SpeakerConfig.DVDAUDIO_GR2_8,
            SpeakerConfig.DVDAUDIO_GR2_9,
            SpeakerConfig.DVDAUDIO_GR2_10,
            SpeakerConfig.DVDAUDIO_GR2_11,
            SpeakerConfig.DVDAUDIO_GR2_12,
            SpeakerConfig.DVDAUDIO_GR2_13,
            SpeakerConfig.DVDAUDIO_GR2_14,
            SpeakerConfig.DVDAUDIO_GR2_15,
            SpeakerConfig.DVDAUDIO_GR2_16,
            SpeakerConfig.DVDAUDIO_GR2_17,
            SpeakerConfig.DVDAUDIO_GR2_18,
            SpeakerConfig.DVDAUDIO_GR2_19,
            SpeakerConfig.DVDAUDIO_GR2_20,
        };

        //int ch_remap[][] = {
        // //  Canonical order: Lf Rf C LFE Ls Rs S
        // /*  0 */ {  0, -1, -1, -1, -1, -1, -1},
        // /*  1 */ {  0,  1, -1, -1, -1, -1, -1},
        // /*  2 */ {  0,  1,  2, -1, -1, -1, -1},
        // /*  3 */ {  0,  1,  2,  3, -1, -1, -1},
        // /*  4 */ {  0,  1,  2, -1, -1, -1, -1},
        // /*  5 */ {  0,  1,  2,  3, -1, -1, -1},
        // /*  6 */ {  0,  1,  2,  3,  4, -1, -1},
        // /*  7 */ {  0,  1,  2, -1, -1, -1, -1},
        // /*  8 */ {  0,  1,  2,  3, -1, -1, -1},
        // /*  9 */ {  0,  1,  2,  3,  4, -1, -1},
        // /* 10 */ {  0,  1,  2,  3, -1, -1, -1},
        // /* 11 */ {  0,  1,  2,  3,  4, -1, -1},
        // /* 12 */ {  0,  1,  2,  3,  4,  5, -1},
        // /* 13 */ {  0,  1,  2,  3, -1, -1, -1},
        // /* 14 */ {  0,  1,  2,  3,  4, -1, -1},
        // /* 15 */ {  0,  1,  2,  3, -1, -1, -1},
        // /* 16 */ {  0,  1,  2,  3,  4, -1, -1},
        // /* 17 */ {  0,  1,  2,  3,  4,  5, -1},
        // /* 18 */ {  0,  1,  3,  4,  2, -1, -1},
        // /* 19 */ {  0,  1,  3,  4,  2, -1, -1},
        // /* 20 */ {  0,  1,  4,  5,  2,  3, -1},
        //};

        public const int BLOCK_SIZE = 2048;
        public const byte PCM_STREAM_ID = 0xa0;
        public const byte MLP_STREAM_ID = 0xa1;
    }
}
