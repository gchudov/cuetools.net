using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Management;
using System.Net;
using System.Xml;
using System.Text;
using CUETools.CDImage;
using CUETools.AccurateRip;
using CUETools.CDRepair;
using Krystalware.UploadHelper;

namespace CUETools.CTDB
{
	public class CUEToolsDB
	{
		const string urlbase = "http://db.cuetools.net";
		string userAgent;

		private CDRepairEncode verify;
		private CDImageLayout toc;
		private HttpStatusCode accResult;
		private string subResult;
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

		public void ContactDB(string userAgent)
		{
			this.userAgent = userAgent;
			this.total = 0;

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase + "/lookup.php?tocid=" + toc.TOCID);
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = userAgent;

			if (uploadHelper.onProgress != null)
				uploadHelper.onProgress(this, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, 0));

			try
			{
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				accResult = resp.StatusCode;

				if (accResult == HttpStatusCode.OK)
				{
					total = 0;
					using (Stream responseStream = resp.GetResponseStream())
					{
						using (XmlTextReader reader = new XmlTextReader(responseStream))
						{
							reader.ReadToFollowing("ctdb");
							if (reader.ReadToDescendant("entry"))
								do
								{
									XmlReader entry = reader.ReadSubtree();
									string crc32 = reader["crc32"];
									string confidence = reader["confidence"];
									string npar = reader["npar"];
									string stride = reader["stride"];
									string id = reader["id"];
									byte[] parity = null;
									CDImageLayout entry_toc = null;

									entry.Read();
									while (entry.Read() && entry.NodeType != XmlNodeType.EndElement)
									{
										if (entry.Name == "parity")
										{
											entry.Read(); // entry.NodeType == XmlNodeType.Text
											parity = Convert.FromBase64String(entry.Value);
											entry.Read();
										}
										else if (entry.Name == "toc")
										{
											string trackcount = entry["trackcount"];
											string audiotracks = entry["audiotracks"];
											string firstaudio = entry["firstaudio"];
											entry.Read();
											string trackoffsets = entry.Value;
											entry.Read();
											entry_toc = new CDImageLayout(int.Parse(trackcount), int.Parse(audiotracks), int.Parse(firstaudio), trackoffsets);
										}
										else
										{
											reader.Skip();
										}
									}
									entry.Close();
									total += int.Parse(confidence);
									entries.Add(new DBEntry(parity, 0, parity.Length, int.Parse(confidence), int.Parse(npar), uint.Parse(crc32, NumberStyles.HexNumber), id, entry_toc));
								} while (reader.ReadToNextSibling("entry"));
							reader.Close();
						}
					}
					if (entries.Count == 0)
						accResult = HttpStatusCode.NotFound;
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}
		}

		public void FetchDB(string url, out HttpStatusCode accResult, out byte[] contents, out int total, List<DBEntry> entries)
		{
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = userAgent;
			contents = null;
			total = 0;

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
							int count = 0, pos = 0;
							do
							{
								if (uploadHelper.onProgress != null)
									uploadHelper.onProgress(url, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, ((double)pos) / resp.ContentLength));
								count = responseStream.Read(buffer, 0, buffer.Length);
								memoryStream.Write(buffer, 0, count);
								pos += count;
							} while (count != 0);
							contents = memoryStream.ToArray();
						}
					}
					Parse(contents, entries, out total);
					if (entries.Count == 0)
						accResult = HttpStatusCode.NoContent;
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}
		}

		static string uuidInfo = null;

		public static string GetUUID()
		{
			if (uuidInfo == null)
			{
				ManagementClass mc = new ManagementClass("Win32_ComputerSystemProduct");
				foreach (ManagementObject mo in mc.GetInstances())
				{
					uuidInfo = mo.Properties["UUID"].Value.ToString();
					break;
				}
			}
			return uuidInfo ?? "unknown";
		}

		public string Confirm(DBEntry entry)
		{
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase + "/confirm.php?tocid=" + toc.TOCID + "&id=" + entry.id);
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = userAgent;
			try
			{
				HttpWebResponse resp = (HttpWebResponse)req.GetResponse();
				if (resp.StatusCode != HttpStatusCode.OK)
					subResult = resp.StatusCode.ToString();
				else
				{
					using (Stream s = resp.GetResponseStream())
					using (StreamReader sr = new StreamReader(s))
						subResult = sr.ReadToEnd();
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					subResult = ((HttpWebResponse)ex.Response).StatusCode.ToString();
				else
					subResult = "unknown error";
			}
			return subResult;
		}

		public string Submit(int confidence, int total, string artist, string title)
		{
			UploadFile[] files = new UploadFile[1];
			MemoryStream newcontents = new MemoryStream();
			using (DBHDR FTYP = new DBHDR(newcontents, "ftyp"))
				FTYP.Write("CTDB");
			using (DBHDR CTDB = new DBHDR(newcontents, "CTDB"))
			{
				using (DBHDR HEAD = CTDB.HDR("HEAD"))
				{
					using (DBHDR TOTL = HEAD.HDR("TOTL")) TOTL.Write(total);
					using (DBHDR VERS = HEAD.HDR("VERS")) VERS.Write(0x100);
					using (DBHDR DATE = HEAD.HDR("DATE")) DATE.Write(DateTime.Now);
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
					if (artist != null && artist != "") using (DBHDR TAG = DISC.HDR("ART ")) TAG.Write(artist);
					if (title != null && title != "") using (DBHDR TAG = DISC.HDR("nam ")) TAG.Write(title);
					using (DBHDR USER = DISC.HDR("USER")) USER.Write(GetUUID());
					using (DBHDR TOOL = DISC.HDR("TOOL")) TOOL.Write(userAgent);
					using (DBHDR TOOL = DISC.HDR("MBID")) TOOL.Write(toc.MusicBrainzId);
					using (DBHDR DATE = DISC.HDR("DATE")) DATE.Write(DateTime.Now);
					using (DBHDR CONF = DISC.HDR("CONF")) CONF.Write(confidence);
					using (DBHDR NPAR = DISC.HDR("NPAR")) NPAR.Write(verify.NPAR);
					using (DBHDR CRC_ = DISC.HDR("CRC ")) CRC_.Write(verify.CRC);
					using (DBHDR PAR_ = DISC.HDR("PAR ")) PAR_.Write(verify.Parity);
				}
			}
			newcontents.Position = 0;
			files[0] = new UploadFile(newcontents, "uploadedfile", "data.bin", "image/binary");
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase + "/submit.php");
			req.Proxy = proxy;
			req.UserAgent = userAgent;
			NameValueCollection form = new NameValueCollection();
			form.Add("tocid", toc.TOCID);
			form.Add("crc32", string.Format("%08X", verify.CRC));
			form.Add("parity", Convert.ToBase64String(verify.Parity, 0, 16));
			form.Add("confidence", confidence.ToString());
			form.Add("trackcount", toc.TrackCount.ToString());
			form.Add("firstaudio", toc.FirstAudio.ToString());
			form.Add("audiotracks", toc.AudioTracks.ToString());
			form.Add("trackoffsets", toc.TrackOffsets);
			form.Add("userid", GetUUID());
			form.Add("agent", userAgent);
			if (artist != null && artist != "") form.Add("artist", artist);
			if (title != null && title != "") form.Add("title", title);
			HttpWebResponse resp = uploadHelper.Upload(req, files, form);
			using (Stream s = resp.GetResponseStream())
			using (StreamReader sr = new StreamReader(s))
				subResult = sr.ReadToEnd();
			return subResult;
		}

		private void Parse(byte[] contents, List<DBEntry> entries, out int total)
		{
			ReadDB rdr = new ReadDB(contents);

			total = 0;
			int end;
			string hdr = rdr.ReadHDR(out end);
			uint magic = rdr.ReadUInt();
			if (hdr != "ftyp" || magic != 0x43544442 || end != rdr.pos)
				throw new Exception("invalid CTDB file");
			hdr = rdr.ReadHDR(out end);
			if (hdr != "CTDB" || end != contents.Length) 
				throw new Exception("invalid CTDB file");
			hdr = rdr.ReadHDR(out end);
			if (hdr != "HEAD") 
				throw new Exception("invalid CTDB file");
			int endHead = end;
			while (rdr.pos < endHead)
			{
				hdr = rdr.ReadHDR(out end);
				if (hdr == "TOTL")
					total = rdr.ReadInt();
				rdr.pos = end;
			}
			rdr.pos = endHead;
			while (rdr.pos < contents.Length)
			{
				hdr = rdr.ReadHDR(out end);
				if (hdr != "DISC")
				{
					rdr.pos = end;
					continue;
				}
				int endDisc = end;
				uint crc = 0;
				int parPos = 0, parLen = 0, conf = 0, npar = 0;
				while (rdr.pos < endDisc)
				{
					hdr = rdr.ReadHDR(out end);
					if (hdr == "PAR ")
					{
						parPos = rdr.pos;
						parLen = end - rdr.pos;
					}
					else if (hdr == "CRC ")
						crc = rdr.ReadUInt();
					else if (hdr == "CONF")
						conf = rdr.ReadInt();
					else if (hdr == "NPAR")
						npar = rdr.ReadInt();
					rdr.pos = end;
				}
				if (parPos != 0 && npar >= 2 && npar <= 16 && conf >= 0)
				//if (parPos != 0 && npar >= 2 && npar <= 16 && conf != 0)
					entries.Add(new DBEntry(contents, parPos, parLen, conf, npar, crc, null, null));
			}
		}

		public void DoVerify()
		{
			foreach (DBEntry entry in entries)
			{
				if (entry.toc.Pregap != toc.Pregap || entry.toc.AudioLength != toc.AudioLength)
				{
					entry.hasErrors = false;
					entry.canRecover = false;
					continue;
				}
				if (!verify.FindOffset(entry.npar, entry.parity, entry.pos, entry.crc, out entry.offset, out entry.hasErrors))
					entry.canRecover = false;
				else if (entry.hasErrors)
				{
					byte[] contents2;
					int total2;
					List<DBEntry> entries2 = new List<DBEntry>();
					FetchDB(string.Format("{0}/repair.php?tocid={1}&id={2}", urlbase, toc.TOCID, entry.id), out entry.httpStatus, out contents2, out total2, entries2);
					if (entry.httpStatus != HttpStatusCode.OK)
						entry.canRecover = false;
					else
					{
						entry.repair = verify.VerifyParity(entries2[0].npar, contents2, entries2[0].pos, entries2[0].len, entry.offset);
						entry.canRecover = entry.repair.CanRecover;
					}
				}
			}
		}

		public void Init(bool encode, AccurateRipVerify ar)
		{
			int npar = 8;
			foreach (DBEntry entry in entries)
				npar = Math.Max(npar, entry.npar);
			verify = new CDRepairEncode(length, 10 * 588 * 2, npar, entries.Count > 0, encode, ar);
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

		public string SubStatus
		{
			get
			{
				return subResult;
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
						if (entry.toc.Pregap == toc.Pregap && (!entry.hasErrors || entry.canRecover))
							if (popular == null || entry.conf > popular.conf)
								popular = entry;
					if (popular != null)
						res = popular.Status;
					foreach (DBEntry entry in entries)
						if (entry != popular && entry.toc.Pregap == toc.Pregap && (!entry.hasErrors || entry.canRecover))
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
		public byte[] parity;
		public int pos;
		public int len;
		public int conf;
		public int npar;
		public int offset;
		public uint crc;
		public bool hasErrors;
		public bool canRecover;
		public CDRepairFix repair;
		public HttpStatusCode httpStatus;
		public string id;
		public CDImageLayout toc;

		public DBEntry(byte[] parity, int pos, int len, int conf, int npar, uint crc, string id, CDImageLayout toc)
		{
			this.parity = parity;
			this.id = id;
			this.pos = pos;
			this.len = len;
			this.conf = conf;
			this.crc = crc;
			this.npar = npar;
			this.toc = toc;
		}

		public string Status
		{
			get
			{
				if (!hasErrors)
					return string.Format("verified OK, confidence {0}", conf);
				if (canRecover)
					return string.Format("differs in {1} samples, confidence {0}", conf, repair.CorrectableErrors);
				if (httpStatus == HttpStatusCode.OK)
					return "could not be verified";
				return "could not be verified: " + httpStatus.ToString();
			}
		}
	}

	internal class ReadDB
	{
		byte[] contents;
		public int pos;

		public ReadDB(byte[] contents)
		{
			this.contents = contents;
			pos = 0;
		}

		public string ReadHDR(out int end)
		{
			int size = ReadInt();
			string res = Encoding.ASCII.GetString(contents, pos, 4);
			pos += 4;
			end = pos + size - 8;
			return res;
		}

		public int ReadInt()
		{
			int value =
				(contents[pos + 3] +
				(contents[pos + 2] << 8) +
				(contents[pos + 1] << 16) +
				(contents[pos + 0] << 24));
			pos += 4;
			return value;
		}

		public uint ReadUInt()
		{
			uint value =
				((uint)contents[pos + 3] +
				((uint)contents[pos + 2] << 8) +
				((uint)contents[pos + 1] << 16) +
				((uint)contents[pos + 0] << 24));
			pos += 4;
			return value;
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
