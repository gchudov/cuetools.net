using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Management;
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
		private string subResult;
		private byte[] contents;
		private int pos;
		private int length;
		private int total;
		List<DBEntry> entries = new List<DBEntry>();
		DBEntry selectedEntry;
		IWebProxy proxy;
		HttpUploadHelper uploadHelper;

		public CUEToolsDB(CDImageLayout toc, IWebProxy proxy)
		{
			this.toc = toc;
			this.length = (int)toc.AudioLength * 588;
			this.proxy = proxy;
			this.uploadHelper = new HttpUploadHelper();
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
			req.Proxy = proxy;

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
				if (entries.Count == 0)
					accResult = HttpStatusCode.NoContent;
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}
		}

		static string cpuInfo = null;

		public static string GetCPUID()
		{
			if (cpuInfo == null)
			{
				ManagementClass mc = new ManagementClass("win32_processor");
				foreach (ManagementObject mo in mc.GetInstances())
				{
					//Get only the first CPU's ID
					cpuInfo = mo.Properties["processorID"].Value.ToString();
					break;
				}
			}
			return cpuInfo ?? "unknown";
		}

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
			using (DBHDR FTYP = new DBHDR(newcontents, "ftyp"))
				FTYP.Write("CTDB");
			using (DBHDR CTDB = new DBHDR(newcontents, "CTDB"))
			{
				using (DBHDR HEAD = CTDB.HDR("HEAD"))
				{
					HEAD.Write(0x100);			// version
					HEAD.Write(1);				// disc count
					HEAD.Write(total);			// total submissions
					HEAD.Write(DateTime.Now);	// date
				}
				using (DBHDR DISC = CTDB.HDR("DISC"))
				{
					using (DBHDR TOC = DISC.HDR("TOC "))
					{
						using (DBHDR INFO = TOC.HDR("INFO"))
						{
							INFO.Write(toc.TrackCount);
							INFO.Write(toc.Pregap);
						}
						for (int i = 1; i <= toc.TrackCount; i++)
							using (DBHDR TRAK = TOC.HDR("TRAK"))
							{
								TRAK.Write(toc[i].IsAudio ? 1 : 0);
								TRAK.Write(toc[i].Length);
							}
					}
					using (DBHDR USER = DISC.HDR("USER")) USER.Write(GetCPUID());
					using (DBHDR TOOL = DISC.HDR("TOOL")) TOOL.Write("CUETools 205");
					using (DBHDR DATE = DISC.HDR("DATE")) DATE.Write(DateTime.Now);
					using (DBHDR CONF = DISC.HDR("CONF")) CONF.Write(confidence);
					using (DBHDR NPAR = DISC.HDR("NPAR")) NPAR.Write(verify.NPAR);
					using (DBHDR CRC_ = DISC.HDR("CRC ")) CRC_.Write(verify.CRC);
					using (DBHDR PAR_ = DISC.HDR("PAR ")) PAR_.Write(verify.Parity);
				}
			}
			newcontents.Position = 0;
			files[0] = new UploadFile(newcontents, "uploadedfile", "data.bin", "image/binary");
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create("http://db.cuetools.net/uploader.php");
			req.Proxy = proxy;
			req.UserAgent = "CUETools 205";
			NameValueCollection form = new NameValueCollection();
			form.Add("id", String.Format("{0:d3}-{1:x8}-{2:x8}-{3:x8}", toc.AudioTracks, discId1, discId2, cddbDiscId));
			HttpWebResponse resp = uploadHelper.Upload(req, files, form);
			using (Stream s = resp.GetResponseStream())
			using (StreamReader sr = new StreamReader(s))
				subResult = sr.ReadToEnd();
			return subResult;
		}

		private string ReadHDR(out int end)
		{
			int size = ReadInt();
			string res = Encoding.ASCII.GetString(contents, pos, 4);
			pos += 4;
			end = pos + size - 8;
			return res;
		}

		private int ReadInt()
		{
			int value =
				(contents[pos + 3] +
				(contents[pos + 2] << 8) +
				(contents[pos + 1] << 16) +
				(contents[pos + 0] << 24));
			pos += 4;
			return value;
		}

		private uint ReadUInt()
		{
			uint value =
				((uint)contents[pos + 3] +
				((uint)contents[pos + 2] << 8) +
				((uint)contents[pos + 1] << 16) +
				((uint)contents[pos + 0] << 24));
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
			uint magic = ReadUInt();
			if (hdr != "ftyp" || magic != 0x43544442 || end != pos)
				throw new Exception("invalid CTDB file");
			hdr = ReadHDR(out end);
			if (hdr != "CTDB" || end != contents.Length) 
				throw new Exception("invalid CTDB file");
			hdr = ReadHDR(out end);
			if (hdr != "HEAD") 
				throw new Exception("invalid CTDB file");
			uint version = ReadUInt();
			int discCount = ReadInt();
			total = ReadInt();
			if (discCount <= 0 || version >= 0x200)
				throw new Exception("invalid CTDB file");
			// date
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
				int parPos = 0, parLen = 0, conf = 0, npar = 0;
				while (pos < endDisc)
				{
					hdr = ReadHDR(out end);
					if (hdr == "PAR ")
					{
						parPos = pos;
						parLen = end - pos;
					}
					else if (hdr == "CRC ")
						crc = ReadUInt();
					else if (hdr == "CONF")
						conf = ReadInt();
					else if (hdr == "NPAR")
						npar = ReadInt();
					pos = end;
				}
				if (parPos != 0 && npar >= 2 && npar <= 16 && conf >= 0)
				//if (parPos != 0 && npar >= 2 && npar <= 16 && conf != 0)
					entries.Add(new DBEntry(parPos, parLen, conf, npar, crc));
			}
		}

		public void DoVerify()
		{
			foreach (DBEntry entry in entries)
			{
				if (!verify.FindOffset(entry.npar, contents, entry.pos, entry.crc, out entry.offset, out entry.hasErrors))
					entry.canRecover = false;
				else if (entry.hasErrors)
				{
					entry.repair = verify.VerifyParity(entry.npar, contents, entry.pos, entry.len, entry.offset);
					entry.canRecover = entry.repair.CanRecover;
				}
			}
		}

		public void Init(bool encode)
		{
			int npar = 8;
			foreach (DBEntry entry in entries)
				npar = Math.Max(npar, entry.npar);
			verify = new CDRepairEncode(length, 10 * 588 * 2, npar, entries.Count > 0, encode);
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

		public DBEntry SelectedEntry
		{
			set
			{
				selectedEntry = value;
			}
			get
			{
				return selectedEntry;
			}
		}

		public string Status
		{
			get
			{
				//sw.WriteLine("CUETools DB CRC: {0:x8}", Verify.CRC);
				string res = null;
				if (DBStatus != null)
					res = DBStatus;
				else
				{
					DBEntry popular = null;
					foreach (DBEntry entry in entries)
						if (!entry.hasErrors || entry.canRecover)
							if (popular == null || entry.conf > popular.conf)
								popular = entry;
					if (popular != null)
						res = popular.Status;
					foreach (DBEntry entry in entries)
						if (entry != popular && (!entry.hasErrors || entry.canRecover))
							res += ", or " + entry.Status;
					if (res == null)
						res = "could not be verified";
				}
				if (subResult != null)
					res += ", " + subResult;
				return res;
			}
		}

		public IEnumerable<DBEntry> Entries
		{
			get
			{
				return entries;
			}
		}

		public HttpUploadHelper UploadHelper
		{
			get
			{
				return uploadHelper;
			}
		}
	}

	public class DBEntry
	{
		public int pos;
		public int len;
		public int conf;
		public int npar;
		public int offset;
		public uint crc;
		public bool hasErrors;
		public bool canRecover;
		public CDRepairFix repair;

		public DBEntry(int pos, int len, int conf, int npar, uint crc)
		{
			this.pos = pos;
			this.len = len;
			this.conf = conf;
			this.crc = crc;
			this.npar = npar;
		}

		public string Status
		{
			get
			{
				if (!hasErrors)
					return string.Format("verified OK, confidence {0}", conf);
				if (canRecover)
					return string.Format("contains {1} correctable errors, confidence {0}", conf, repair.CorrectableErrors);
				return "could not be verified";
			}
		}
	}

	internal class DBHDR : IDisposable
	{
		private long lenOffs;
		private MemoryStream stream;

		public DBHDR(MemoryStream stream, string name)
		{
			this.stream = stream;
			lenOffs = stream.Position;
			Write(0);
			Write(name);
		}

		public void Dispose()
		{
			long fin = stream.Position;
			stream.Position = lenOffs;
			Write((int)(fin - lenOffs));
			stream.Position = fin;
		}

		public void Write(int value)
		{
			byte[] temp = new byte[4];
			temp[3] = (byte)(value & 0xff);
			temp[2] = (byte)((value >> 8) & 0xff);
			temp[1] = (byte)((value >> 16) & 0xff);
			temp[0] = (byte)((value >> 24) & 0xff);
			Write(temp);
		}

		public void Write(uint value)
		{
			byte[] temp = new byte[4];
			temp[3] = (byte)(value & 0xff);
			temp[2] = (byte)((value >> 8) & 0xff);
			temp[1] = (byte)((value >> 16) & 0xff);
			temp[0] = (byte)((value >> 24) & 0xff);
			Write(temp);
		}

		public void Write(long value)
		{
			byte[] temp = new byte[8];
			temp[7] = (byte)((value) & 0xff);
			temp[6] = (byte)((value >> 8) & 0xff);
			temp[5] = (byte)((value >> 16) & 0xff);
			temp[4] = (byte)((value >> 24) & 0xff);
			temp[3] = (byte)((value >> 32) & 0xff);
			temp[2] = (byte)((value >> 40) & 0xff);
			temp[1] = (byte)((value >> 48) & 0xff);
			temp[0] = (byte)((value >> 56) & 0xff);
			Write(temp);
		}

		public void Write(string value)
		{
			Write(Encoding.UTF8.GetBytes(value));
		}

		public void Write(DateTime value)
		{
			Write(value.ToFileTimeUtc());
		}
		
		public void Write(byte[] value)
		{
			stream.Write(value, 0, value.Length);
		}

		public DBHDR HDR(string name)
		{
			return new DBHDR(stream, name);
		}
	}
}
