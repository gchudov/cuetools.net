using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text;
using CUETools.CDImage;
using CUETools.CDRepair;
using Krystalware.UploadHelper;

namespace CUETools.CTDB
{
	public class CUEToolsDB
	{
		private CDRepairEncode verify;
		private CDImageLayout toc;
		private HttpStatusCode accResult;
		private string id;
		private byte[] contents;
		private int pos;
		private int length;
		private int confidence;
		private int total;
		List<DBEntry> entries = new List<DBEntry>();

		public CUEToolsDB(CDImageLayout toc)
		{
			this.toc = toc;
			this.length = (int)toc.AudioLength * 588;
		}

		public void ContactDB(string id)
		{
			this.id = id;

			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;

			string[] n = id.Split('-');
			if (n.Length != 3)
				throw new Exception("Invalid accurateRipId.");
			discId1 = UInt32.Parse(n[0], NumberStyles.HexNumber);
			discId2 = UInt32.Parse(n[1], NumberStyles.HexNumber);
			cddbDiscId = UInt32.Parse(n[2], NumberStyles.HexNumber);

			string url = String.Format("http://db.cuetools.net/parity/{0:x}/{1:x}/{2:x}/dBCT-{3:d3}-{4:x8}-{5:x8}-{6:x8}.bin",
				discId1 & 0xF, discId1 >> 4 & 0xF, discId1 >> 8 & 0xF, toc.AudioTracks, discId1, discId2, cddbDiscId);

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.Method = "GET";
			req.Proxy = WebRequest.GetSystemWebProxy();

			try
			{
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				accResult = resp.StatusCode;

				if (accResult == HttpStatusCode.OK)
				{
					using (Stream responseStream = resp.GetResponseStream())
					{
						using(MemoryStream memoryStream = new MemoryStream())
						{
							byte[] buffer = new byte[16536];
							int count = 0;
							do
							{
								count = responseStream.Read(buffer, 0, buffer.Length);
								memoryStream.Write(buffer, 0, count);
							} while (count != 0);
							contents = memoryStream.ToArray();
						}
					}
				}				
				Parse();
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}
		}

		/// <summary>
		/// Database entry format:
		/// 'CTDB' version(0x00000100) size
		/// 'HEAD' version(0x00000100) size disc-count total-submissions
		/// 'DISC' version(0x00000100) size
		///   'TOC ' version(0x00000100) size track-count preGap
		///     'TRAK' ersion(0x00000100) size flags(1==isAudio) length(frames)
		///      .... track-count
		///   'ARDB' version(0x00000100) size pressings-count
		///     'ARCD' version(0x00000100) size CRC1 ... CRCN
		///      .... pressings-count
		///   'PAR8' version(0x00000100) size parity-data
		///   'CONF' version(0x00000100) size confidence
		///   'CRC ' version(0x00000100) size CRC32 min-offset max-offset ...
		///   .... disc-count
		/// </summary>
		public string Submit(int confidence, int total)
		{
			if (id == null)
				throw new Exception("no id");
			// Calculate the three disc ids used by AR
			uint discId1 = 0;
			uint discId2 = 0;
			uint cddbDiscId = 0;

			string[] n = id.Split('-');
			if (n.Length != 3)
				throw new Exception("Invalid accurateRipId.");
			discId1 = UInt32.Parse(n[0], NumberStyles.HexNumber);
			discId2 = UInt32.Parse(n[1], NumberStyles.HexNumber);
			cddbDiscId = UInt32.Parse(n[2], NumberStyles.HexNumber);

			UploadFile[] files = new UploadFile[1];
			MemoryStream newcontents = new MemoryStream();
			using (DBHDR CTDB = new DBHDR(newcontents, "CTDB", 0x100))
			{
				using (DBHDR HEAD = new DBHDR(newcontents, "HEAD", 0x100))
				{
					DBHDR.WriteInt(newcontents, 1); // disc count
					DBHDR.WriteInt(newcontents, total);
				}
				using (DBHDR DISC = new DBHDR(newcontents, "DISC", 0x100))
				{
					using (DBHDR TOC = new DBHDR(newcontents, "TOC ", 0x100))
					{
						DBHDR.WriteInt(newcontents, toc.TrackCount);
						DBHDR.WriteInt(newcontents, toc.Pregap);
						for (int i = 1; i <= toc.TrackCount; i++)
							using (DBHDR TRAK = new DBHDR(newcontents, "TRAK", 0x100))
							{
								DBHDR.WriteInt(newcontents, toc[i].IsAudio ? 1 : 0);
								DBHDR.WriteInt(newcontents, toc[i].Length);
							}
					}
					using (DBHDR PAR8 = new DBHDR(newcontents, "CONF", 0x100))
					{
						DBHDR.WriteInt(newcontents, confidence);
					}
					using (DBHDR PAR8 = new DBHDR(newcontents, "CRC ", 0x100))
					{
						DBHDR.WriteInt(newcontents, verify.CRC);
					}
					using (DBHDR PAR8 = new DBHDR(newcontents, "PAR8", 0x100))
					{
						newcontents.Write(verify.Parity, 0, verify.Parity.Length);
					}
				}
			}
			newcontents.Position = 0;
			files[0] = new UploadFile(newcontents, "uploadedfile", "data.bin", "image/binary");
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://db.cuetools.net/uploader.php");
			req.Proxy = WebRequest.GetSystemWebProxy();
			NameValueCollection form = new NameValueCollection();
			form.Add("id", String.Format("{0:d3}-{1:x8}-{2:x8}-{3:x8}", toc.AudioTracks, discId1, discId2, cddbDiscId));
			HttpWebResponse resp = HttpUploadHelper.Upload(req, files, form);
			string errtext;
			using (Stream s = resp.GetResponseStream())
			using (StreamReader sr = new StreamReader(s))
			{
				errtext = sr.ReadToEnd();
			}
			return errtext;
		}

		private string ReadHDR(out int end)
		{
			string res = Encoding.ASCII.GetString(contents, pos, 4);
			pos += 4;
			int version = ReadInt();
			if (version >= 0x200)
				throw new Exception("unsupported CTDB version");
			int size = ReadInt();
			end = pos + size;
			return res;
		}

		private int ReadInt()
		{
			int value =
				(contents[pos] +
				(contents[pos + 1] << 8) +
				(contents[pos + 2] << 16) +
				(contents[pos + 3] << 24));
			pos += 4;
			return value;
		}

		private uint ReadUInt()
		{
			uint value =
				((uint)contents[pos] +
				((uint)contents[pos + 1] << 8) +
				((uint)contents[pos + 2] << 16) +
				((uint)contents[pos + 3] << 24));
			pos += 4;
			return value;
		}

		private void Parse()
		{
			if (accResult != HttpStatusCode.OK)
				return;

			pos = 0;
			int end;
			string hdr = ReadHDR(out end);
			if (hdr != "CTDB") throw new Exception("invalid CTDB file");
			if (end != contents.Length) throw new Exception("incomplete CTDB file");
			hdr = ReadHDR(out end);
			if (hdr != "HEAD") throw new Exception("invalid CTDB file");
			int discCount = ReadInt();
			total = ReadInt();
			pos = end;
			while (pos < contents.Length)
			{
				hdr = ReadHDR(out end);
				if (hdr != "DISC")
				{
					pos = end;
					continue;
				}
				int endDisc = end;
				uint crc = 0;
				int parPos = 0, parLen = 0, conf = 0;
				while (pos < endDisc)
				{
					hdr = ReadHDR(out end);
					if (hdr == "PAR8")
					{
						parPos = pos;
						parLen = end - pos;
					}
					else if (hdr == "CRC ")
						crc = ReadUInt();
					else if (hdr == "CONF")
						conf = ReadInt();
					pos = end;
				}
				if (parPos != 0)
					entries.Add(new DBEntry(parPos, parLen, conf, crc));
			}
		}

		public void DoVerify()
		{
			foreach (DBEntry entry in entries)
			{
				verify.VerifyParity(contents, entry.pos, entry.len);
				if (!verify.HasErrors || verify.CanRecover)
					confidence = entry.conf;
				break;
			}
		}

		public void Init()
		{
			verify = new CDRepairEncode(length, 10 * 588 * 2, 8, accResult == HttpStatusCode.OK);
		}

		public int Total
		{
			get
			{
				return total;
			}
		}

		public HttpStatusCode AccResult
		{
			get
			{
				return accResult;
			}
		}

		public CDRepairEncode Verify
		{
			get
			{
				return verify;
			}
		}

		public string DBStatus
		{
			get
			{
				return accResult == HttpStatusCode.NotFound ? "disk not present in database" :
					accResult == HttpStatusCode.OK ? null
					: accResult.ToString();
			}
		}

		public string Status
		{
			get
			{
				//sw.WriteLine("CUETools DB CRC: {0:x8}", Verify.CRC);
				if (DBStatus != null)
					return DBStatus;
				if (!verify.HasErrors)
					return string.Format("verified OK, confidence {0}/{1}", confidence, total);
				if (verify.CanRecover)
					return string.Format("contains correctable errors, confidence {0}/{1}", confidence, total);
				return "could not be verified";
			}
		}
	}

	internal class DBEntry
	{
		public int pos;
		public int len;
		public int conf;
		public uint crc;

		public DBEntry(int pos, int len, int conf, uint crc)
		{
			this.pos = pos;
			this.len = len;
			this.conf = conf;
			this.crc = crc;
		}
	}

	internal class DBHDR : IDisposable
	{
		private long lenOffs;
		private MemoryStream stream;

		public DBHDR(MemoryStream stream, string name, int version)
		{
			this.stream = stream;
			stream.Write(Encoding.ASCII.GetBytes(name), 0, 4);
			WriteInt(stream, version);
			lenOffs = stream.Position;
			WriteInt(stream, 0);
		}

		public void Dispose()
		{
			long fin = stream.Position;
			stream.Position = lenOffs;
			WriteInt(stream, (int)(fin - lenOffs - 4));
			stream.Position = fin;
		}

		public static void WriteInt(MemoryStream stream, int value)
		{
			byte[] temp = new byte[4];
			temp[0] = (byte)(value & 0xff);
			temp[1] = (byte)((value >> 8) & 0xff);
			temp[2] = (byte)((value >> 16) & 0xff);
			temp[3] = (byte)((value >> 24) & 0xff);
			stream.Write(temp, 0, 4);
		}

		public static void WriteInt(MemoryStream stream, uint value)
		{
			byte[] temp = new byte[4];
			temp[0] = (byte)(value & 0xff);
			temp[1] = (byte)((value >> 8) & 0xff);
			temp[2] = (byte)((value >> 16) & 0xff);
			temp[3] = (byte)((value >> 24) & 0xff);
			stream.Write(temp, 0, 4);
		}
	}
}
