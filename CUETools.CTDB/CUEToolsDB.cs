using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Management;
using System.Net;
using System.Xml;
using System.Xml.Serialization;
using System.Text;
using System.Security.Cryptography;
using CUETools.CDImage;
using CUETools.AccurateRip;
using Krystalware.UploadHelper;

namespace CUETools.CTDB
{
	public class CUEToolsDB
	{
		const string urlbase = "http://db.cuetools.net";
		string userAgent;
		string driveName;

		private CDRepairEncode verify;
		private CDImageLayout toc;
		private string subResult;
		private int length;
		private int total;
		private List<DBEntry> entries = new List<DBEntry>();
		private List<CTDBResponseMeta> metadata = new List<CTDBResponseMeta>();
		private DBEntry selectedEntry;
		private IWebProxy proxy;
		private HttpUploadHelper uploadHelper;
		private HttpWebRequest currentReq;
		private int connectTimeout, socketTimeout;

		public CUEToolsDB(CDImageLayout toc, IWebProxy proxy)
		{
			this.toc = toc;
			this.length = (int)toc.AudioLength * 588;
			this.proxy = proxy;
			this.uploadHelper = new HttpUploadHelper();
			this.QueryExceptionStatus = WebExceptionStatus.Pending;
			this.connectTimeout = 15000;
			this.socketTimeout = 30000;
		}

		public void CancelRequest()
		{
			var r = currentReq;
			// copy link to currentReq, because it can be set to null by other thread.
			if (r != null)
			{
				r.Abort();
			}
		}

		public void ContactDB(string userAgent, string driveName, bool musicbrainz, bool fuzzy)
		{
			this.driveName = driveName;
			this.userAgent = userAgent + " (" + Environment.OSVersion.VersionString + ")" + (driveName != null ? " (" + driveName + ")" : "");
			this.total = 0;

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase
				+ "/lookup2.php?musicbrainz=" + (musicbrainz ? 1 : 0) 
				+ "&fuzzy=" + (fuzzy ? 1 : 0) 
				+ "&toc=" + toc.ToString());
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = this.userAgent;
			req.Timeout = connectTimeout;
			req.ReadWriteTimeout = socketTimeout;

			if (uploadHelper.onProgress != null)
				uploadHelper.onProgress(this, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, 0));

			currentReq = req;
			try
			{
				using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
				{
					this.QueryExceptionStatus = WebExceptionStatus.ProtocolError;
					this.QueryResponseStatus = resp.StatusCode;
					if (this.QueryResponseStatus == HttpStatusCode.OK)
					{
						XmlSerializer serializer = new XmlSerializer(typeof(CTDBResponse));
						this.total = 0;
						using (Stream responseStream = resp.GetResponseStream())
						{
							CTDBResponse ctdbResp = serializer.Deserialize(responseStream) as CTDBResponse;
							if (ctdbResp.entry != null)
								foreach (var ctdbRespEntry in ctdbResp.entry)
								{
									if (ctdbRespEntry.toc == null)
										continue;

									var parity = Convert.FromBase64String(ctdbRespEntry.parity);
									var entry_toc = CDImageLayout.FromString(ctdbRespEntry.toc);
									this.total += ctdbRespEntry.confidence;
									var entry = new DBEntry(
										parity,
										0,
										parity.Length,
										ctdbRespEntry.confidence,
										ctdbRespEntry.npar,
										ctdbRespEntry.stride,
										uint.Parse(ctdbRespEntry.crc32, NumberStyles.HexNumber),
										ctdbRespEntry.id,
										entry_toc,
										ctdbRespEntry.hasparity);
									entries.Add(entry);
								}
							if (ctdbResp.musicbrainz != null && ctdbResp.musicbrainz.Length != 0)
								metadata.AddRange(ctdbResp.musicbrainz);
						}
						if (entries.Count == 0)
							this.QueryResponseStatus = HttpStatusCode.NotFound;
						else
							this.QueryExceptionStatus = WebExceptionStatus.Success;
					}
				}
			}
			catch (WebException ex)
			{
				this.QueryExceptionStatus = ex.Status;
				this.QueryExceptionMessage = ex.Message;
				if (this.QueryExceptionStatus == WebExceptionStatus.ProtocolError)
					this.QueryResponseStatus = (ex.Response as HttpWebResponse).StatusCode;
			}
			catch (Exception ex)
			{
				this.QueryExceptionStatus = WebExceptionStatus.UnknownError;
				this.QueryExceptionMessage = ex.Message;
			}
			finally
			{
				currentReq = null;
			}
		}

		public void FetchDB(DBEntry entry)
		{
			string url = entry.hasParity[0] == '/' ? urlbase + entry.hasParity : entry.hasParity;
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = this.userAgent;
			req.Timeout = connectTimeout;
			req.ReadWriteTimeout = socketTimeout;
			req.AutomaticDecompression = DecompressionMethods.None;

			if (uploadHelper.onProgress != null)
				uploadHelper.onProgress(url, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, 0.0));

			currentReq = req;
			try
			{
				using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
				{
					entry.httpStatus = resp.StatusCode;

					if (entry.httpStatus == HttpStatusCode.OK)
					{
						if (resp.ContentLength < entry.npar * entry.stride * 4 ||
							resp.ContentLength > entry.npar * entry.stride * 8)
						{
							entry.httpStatus = HttpStatusCode.PartialContent;
						}
					}

					if (entry.httpStatus == HttpStatusCode.OK)
					{
						using (Stream responseStream = resp.GetResponseStream())
						{
							byte[] contents = new byte[resp.ContentLength];
							int pos = 0, count = 0;
							do
							{
								count = responseStream.Read(contents, pos, Math.Min(contents.Length - pos, 32768));
								pos += count;
								if (uploadHelper.onProgress != null)
									uploadHelper.onProgress(url, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, ((double)pos) / contents.Length));
							} while (count != 0);
							if (!Parse(contents, entry))
								entry.httpStatus = HttpStatusCode.NoContent;						
						}
					}
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					entry.httpStatus = ((HttpWebResponse)ex.Response).StatusCode;
				else
					entry.httpStatus = HttpStatusCode.BadRequest;
			}
			finally
			{
				currentReq = null;
			}
		}

		static string uuidInfo = null;

		public static string GetUUID()
		{
			if (uuidInfo == null)
			{
				string id = "CTDB userid";
				using (ManagementClass mc = new ManagementClass("Win32_ComputerSystemProduct"))
					foreach (ManagementObject mo in mc.GetInstances())
					{
						id = id + mo.Properties["UUID"].Value.ToString();
						break;
					}
				byte[] hashBytes = (new SHA1CryptoServiceProvider()).ComputeHash(Encoding.ASCII.GetBytes(id));
				uuidInfo = Convert.ToBase64String(hashBytes).Replace('+', '.').Replace('/', '_').Replace('=', '-');
			}
			return uuidInfo;
		}

		public string Submit(int confidence, int quality, string artist, string title)
		{
			if (this.QueryExceptionStatus != WebExceptionStatus.Success &&
				(this.QueryExceptionStatus != WebExceptionStatus.ProtocolError || this.QueryResponseStatus != HttpStatusCode.NotFound))
				return this.DBStatus;
			DBEntry confirm = null;
			foreach (DBEntry entry in this.Entries)
				if (entry.toc.TrackOffsets == this.toc.TrackOffsets && !entry.hasErrors)
					confirm = entry;
			if (confirm != null) confidence = 1;
			DoSubmit(confidence, quality, artist, title, false, confirm);
			if (subResult == "parity needed")
				DoSubmit(confidence, quality, artist, title, true, confirm);
			return subResult;
		}

		protected string DoSubmit(int confidence, int quality, string artist, string title, bool upload, DBEntry confirm)
		{
			UploadFile[] files;
			if (upload)
			{
				MemoryStream newcontents = new MemoryStream();
				using (DBHDR FTYP = new DBHDR(newcontents, "ftyp"))
					FTYP.Write("CTDB");
				using (DBHDR CTDB = new DBHDR(newcontents, "CTDB"))
				{
					using (DBHDR HEAD = CTDB.HDR("HEAD"))
					{
						using (DBHDR VERS = HEAD.HDR("VERS")) VERS.Write(0x101);
					}
					using (DBHDR DISC = CTDB.HDR("DISC"))
					{
						using (DBHDR CONF = DISC.HDR("CONF")) CONF.Write(confidence);
						using (DBHDR NPAR = DISC.HDR("NPAR")) NPAR.Write(verify.NPAR);
						using (DBHDR CRC_ = DISC.HDR("CRC ")) CRC_.Write(verify.CRC);
						using (DBHDR PAR_ = DISC.HDR("PAR ")) PAR_.Write(verify.Parity);
					}
				}
				newcontents.Position = 0;
				files = new UploadFile[1] { new UploadFile(newcontents, "parityfile", "data.bin", "image/binary") };
			}
			else
			{
				files = new UploadFile[0];
			}
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase + "/submit2.php");
			req.Proxy = proxy;
			req.UserAgent = this.userAgent;
			req.Timeout = connectTimeout;
			req.ReadWriteTimeout = socketTimeout;
			NameValueCollection form = new NameValueCollection();
			if (upload)
				form.Add("parityfile", "1");
			if (confirm != null)
				form.Add("confirmid", confirm.id);
			form.Add("toc", toc.ToString());
			form.Add("crc32", ((int)verify.CRC).ToString());
			form.Add("trackcrcs", verify.TrackCRCs);
			form.Add("parity", Convert.ToBase64String(verify.Parity, 0, 16));
			form.Add("confidence", confidence.ToString());
			form.Add("userid", GetUUID());
			form.Add("quality", quality.ToString());
			if (driveName != null)
				form.Add("drivename", driveName);
			if (artist != null && artist != "") form.Add("artist", artist);
			if (title != null && title != "") form.Add("title", title);

			var ExceptionStatus = WebExceptionStatus.Pending;
			string ExceptionMessage = null;
			HttpStatusCode ResponseStatus = HttpStatusCode.OK;
			currentReq = req;
			try
			{
				using (HttpWebResponse resp = uploadHelper.Upload(req, files, form))
				{
					ExceptionStatus = WebExceptionStatus.ProtocolError;
					ResponseStatus = resp.StatusCode;
					if (ResponseStatus == HttpStatusCode.OK)
					{
						ExceptionStatus = WebExceptionStatus.Success;
						using (Stream s = resp.GetResponseStream())
						using (StreamReader sr = new StreamReader(s))
							subResult = sr.ReadToEnd();
						return subResult;
					}
				}
			}
			catch (WebException ex)
			{
				ExceptionStatus = ex.Status;
				ExceptionMessage = ex.Message;
				if (ExceptionStatus == WebExceptionStatus.ProtocolError)
					ResponseStatus = (ex.Response as HttpWebResponse).StatusCode;
			}
			finally
			{
				currentReq = null;
			}
			subResult = ExceptionStatus == WebExceptionStatus.Success ? null :
				ExceptionStatus != WebExceptionStatus.ProtocolError ? ("database access error: " + (ExceptionMessage ?? ExceptionStatus.ToString())) :
				ResponseStatus != HttpStatusCode.NotFound ? "database access error: " + ResponseStatus.ToString() :
				"disk not present in database";
			return subResult;
		}

		private bool Parse(byte[] contents, DBEntry entry)
		{
			if (contents.Length == entry.npar * entry.stride * 4)
			{
				entry.parity = contents;
				entry.pos = 0;
				entry.len = contents.Length;
				return true;
			}

			ReadDB rdr = new ReadDB(contents);

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
				int parPos = 0, parLen = 0;
				while (rdr.pos < endDisc)
				{
					hdr = rdr.ReadHDR(out end);
					if (hdr == "PAR ")
					{
						parPos = rdr.pos;
						parLen = end - rdr.pos;
					}
					rdr.pos = end;
				}
				if (parPos != 0)
				{
					entry.parity = contents;
					entry.pos = parPos;
					entry.len = parLen;
					return true;
				}
			}
			return false;
		}

		public void DoVerify()
		{
			if (this.QueryExceptionStatus != WebExceptionStatus.Success)
				return;
			foreach (DBEntry entry in entries)
			{
				if (entry.toc.Pregap != toc.Pregap || entry.toc.AudioLength != toc.AudioLength || entry.stride != verify.Stride / 2)
				{
					entry.hasErrors = true;
					entry.canRecover = false;
					continue;
				}
				if (!verify.FindOffset(entry.npar, entry.parity, entry.pos, entry.crc, out entry.offset, out entry.hasErrors))
					entry.canRecover = false;
				else if (entry.hasErrors)
				{
					if (entry.hasParity == null || entry.hasParity == "")
						entry.canRecover = false;
					else
					{
						FetchDB(entry);
						if (entry.httpStatus != HttpStatusCode.OK)
							entry.canRecover = false;
						else
						{
							entry.repair = verify.VerifyParity(entry.npar, entry.parity, entry.pos, entry.len, entry.offset);
							entry.canRecover = entry.repair.CanRecover;
						}
					}
				}
			}
		}

		public void Init(bool encode, AccurateRipVerify ar)
		{
			int npar = 8;
			foreach (DBEntry entry in entries)
				npar = Math.Max(npar, entry.npar);
			verify = new CDRepairEncode(ar, 10 * 588 * 2, npar);
		}

		public CDImageLayout TOC
		{
			get
			{
				return toc;
			}
			set
			{
				toc = value;
			}
		}

		public int Total
		{
			get
			{
				return total;
			}
		}

		public WebExceptionStatus QueryExceptionStatus { get; private set; }

		public string QueryExceptionMessage { get; private set; }

		public HttpStatusCode QueryResponseStatus { get; private set; }

		public string DBStatus
		{
			get
			{
				return QueryExceptionStatus == WebExceptionStatus.Success ? null :
					QueryExceptionStatus != WebExceptionStatus.ProtocolError ? ("database access error: " + (QueryExceptionMessage ?? QueryExceptionStatus.ToString())) :
					QueryResponseStatus != HttpStatusCode.NotFound ? "database access error: " + QueryResponseStatus.ToString() :
					"disk not present in database";
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
			set
			{
				subResult = value;
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

		public IEnumerable<CTDBResponseMeta> Metadata
		{
			get
			{
				return metadata;
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
		public int stride;
		public int offset;
		public uint crc;
		public bool hasErrors;
		public bool canRecover;
		public CDRepairFix repair;
		public HttpStatusCode httpStatus;
		public string id;
		public CDImageLayout toc;
		public string hasParity;

		public DBEntry(byte[] parity, int pos, int len, int conf, int npar, int stride, uint crc, string id, CDImageLayout toc, string hasParity)
		{
			this.parity = parity;
			this.id = id;
			this.pos = pos;
			this.len = len;
			this.conf = conf;
			this.crc = crc;
			this.npar = npar;
			this.stride = stride;
			this.toc = toc;
			this.hasParity = hasParity;
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

	[Serializable]
	public class CTDBResponseEntry
	{
		[XmlAttribute]
		public string id { get; set; }
		[XmlAttribute]
		public string crc32 { get; set; }
		[XmlAttribute]
		public int confidence { get; set; }
		[XmlAttribute]
		public int npar { get; set; }
		[XmlAttribute]
		public int stride { get; set; }
		[XmlAttribute]
		public string hasparity { get; set; }
		[XmlAttribute]
		public string parity { get; set; }
		[XmlAttribute]
		public string toc { get; set; }
	}

	[Serializable]
	public class CTDBResponseMetaTrack
	{
		[XmlAttribute]
		public string name { get; set; }
		[XmlAttribute]
		public string artist { get; set; }
	}

	[Serializable]
	public class CTDBResponseMetaLabel
	{
		[XmlAttribute]
		public string name { get; set; }
		[XmlAttribute]
		public string catno { get; set; }
	}
	
	[Serializable]
	public class CTDBResponseMeta
	{
		[XmlAttribute]
		public string release_gid { get; set; }
		[XmlAttribute]
		public string artist { get; set; }
		[XmlAttribute]
		public string album { get; set; }
		[XmlAttribute]
		public string year { get; set; }
		[XmlAttribute]
		public string genre { get; set; }
		[XmlAttribute]
		public string country { get; set; }
		[XmlAttribute]
		public string releasedate { get; set; }
		[XmlAttribute]
		public string discnumber { get; set; }
		[XmlAttribute]
		public string disccount { get; set; }
		[XmlAttribute]
		public string discname { get; set; }
		[XmlAttribute]
		public string coverarturl { get; set; }
		[XmlAttribute]
		public string infourl { get; set; }
		[XmlAttribute]
		public string barcode { get; set; }
		[XmlElement]
		public CTDBResponseMetaTrack[] track;
		[XmlElement]
		public CTDBResponseMetaLabel[] label;
	}

	[Serializable]
	[XmlRoot(ElementName="ctdb", Namespace="http://db.cuetools.net/ns/mmd-1.0#")]
	public class CTDBResponse
	{
		[XmlElement]
		public CTDBResponseEntry[] entry;
		[XmlElement]
		public CTDBResponseMeta[] musicbrainz;
	}
}
