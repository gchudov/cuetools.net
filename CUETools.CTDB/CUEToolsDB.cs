using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.IO;
using System.Management;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Xml.Serialization;
using CUETools.AccurateRip;
using CUETools.CDImage;
using Krystalware.UploadHelper;

namespace CUETools.CTDB
{
	public class CUEToolsDB
	{
		const string defaultServer = "db.cuetools.net";
		string urlbase;
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

		public void ContactDB(string server, string userAgent, string driveName, bool ctdb, bool fuzzy, CTDBMetadataSearch metadataSearch)
		{
			this.driveName = driveName;
			this.userAgent = userAgent + " (" + Environment.OSVersion.VersionString + ")" + (driveName != null ? " (" + driveName + ")" : "");
			this.urlbase = "http://" + (server ?? defaultServer);
			this.total = 0;

			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase
				+ "/lookup2.php"
				+ "?ctdb=" + (ctdb ? "1" : "0")
				+ "&fuzzy=" + (fuzzy ? 1 : 0)
				+ "&metadata=" + (metadataSearch == CTDBMetadataSearch.None ? "none" : metadataSearch == CTDBMetadataSearch.Fast ? "fast" : metadataSearch == CTDBMetadataSearch.Default ? "default" : "extensive")
				+ "&toc=" + toc.ToString());
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = this.userAgent;
			req.Timeout = connectTimeout;
			req.ReadWriteTimeout = socketTimeout;
			req.AutomaticDecompression = DecompressionMethods.Deflate | DecompressionMethods.GZip;

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

		public string Submit(int confidence, int quality, string artist, string title, string barcode)
		{
			if (this.QueryExceptionStatus != WebExceptionStatus.Success &&
				(this.QueryExceptionStatus != WebExceptionStatus.ProtocolError || this.QueryResponseStatus != HttpStatusCode.NotFound))
				return this.DBStatus;
			DBEntry confirm = this.MatchingEntry;
			if (confirm != null) confidence = 1;
			DoSubmit(confidence, quality, artist, title, barcode, false, confirm);
			if (subResult == "parity needed")
				DoSubmit(confidence, quality, artist, title, barcode, true, confirm);
			return subResult;
		}

		protected string DoSubmit(int confidence, int quality, string artist, string title, string barcode, bool upload, DBEntry confirm)
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
			if (driveName != null && driveName != "") form.Add("drivename", driveName);
			if (barcode != null && barcode != "") form.Add("barcode", barcode);
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

		public int Confidence
		{
			get
			{
				if (this.QueryExceptionStatus != WebExceptionStatus.Success)
					return 0;
				int res = 0;
				foreach (DBEntry entry in this.Entries)
					if (entry.toc.ToString() == this.toc.ToString() && !entry.hasErrors)
						res += entry.conf;
				return res;
			}
		}

		public DBEntry MatchingEntry
		{
			get
			{
				if (this.QueryExceptionStatus != WebExceptionStatus.Success)
					return null;
				foreach (DBEntry entry in this.Entries)
					if (entry.toc.ToString() == this.toc.ToString() && !entry.hasErrors)
						return entry;
				return null;
			}
		}

		public void Init(AccurateRipVerify ar)
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
}
