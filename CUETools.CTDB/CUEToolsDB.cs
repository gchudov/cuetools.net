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
using CUETools.Parity;
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
				+ "?ctdb=" + (ctdb ? "2" : "0")
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
                                    this.total += ctdbRespEntry.confidence;
                                    var entry = new DBEntry(ctdbRespEntry);
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

		public ushort[,] FetchDB(DBEntry entry, int npar, ushort[,] syn)
		{
			string url = entry.hasParity[0] == '/' ? urlbase + entry.hasParity : entry.hasParity;
			HttpWebRequest req = (HttpWebRequest)WebRequest.Create(url);
            int prevLen = syn == null ? 0 : syn.GetLength(1) * entry.stride * 2;
			req.Method = "GET";
			req.Proxy = proxy;
			req.UserAgent = this.userAgent;
			req.Timeout = connectTimeout;
			req.ReadWriteTimeout = socketTimeout;
			req.AutomaticDecompression = DecompressionMethods.None;
            req.AddRange(prevLen, npar * entry.stride * 2 - 1);

			currentReq = req;
            try
            {
                using (HttpWebResponse resp = (HttpWebResponse)req.GetResponse())
                {
                    if (resp.StatusCode != HttpStatusCode.OK && resp.StatusCode != HttpStatusCode.PartialContent)
                    {
                        entry.httpStatus = resp.StatusCode;
                        return null;
                    }
                    if (resp.StatusCode == HttpStatusCode.OK && resp.ContentLength == entry.Npar * entry.stride * 2)
                    {
                        npar = entry.Npar;
                        prevLen = 0;
                        syn = null;
                    }
                    else if (resp.StatusCode != HttpStatusCode.PartialContent || (resp.ContentLength + prevLen) != npar * entry.stride * 2)
                    {
                        entry.httpStatus = HttpStatusCode.PartialContent;
                        return null;
                    }

                    using (Stream responseStream = resp.GetResponseStream())
                    {
                        byte[] contents = syn == null ? null : ParityToSyndrome.Syndrome2Bytes(syn);
                        Array.Resize(ref contents, prevLen + (int)resp.ContentLength);
                        int pos = prevLen, count = 0;
                        do
                        {
                            if (uploadHelper.onProgress != null)
                                uploadHelper.onProgress(url, new UploadProgressEventArgs(req.RequestUri.AbsoluteUri, ((double)pos) / (entry.Npar * entry.stride * 2)));
                            count = responseStream.Read(contents, pos, Math.Min(contents.Length - pos, 32768));
                            pos += count;
                        } while (count != 0);

                        if (pos != contents.Length)
                        {
                            entry.httpStatus = HttpStatusCode.PartialContent;
                            return null;
                        }

                        syn = ParityToSyndrome.Bytes2Syndrome(entry.stride, npar, contents);
                        for (int i = 0; i < npar; i++)
                            if (syn[0, i] != entry.syndrome[0, i])
                            {
                                entry.httpStatus = HttpStatusCode.Conflict;
                                return null;
                            }
                        entry.httpStatus = HttpStatusCode.OK;
                        return syn;
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
            return null;
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
            CTDBSubmitResponse resp = null;
            subResult = "";
            var confirms = this.MatchingEntries;
            if (confirms.Count > 0)
            {
                confidence = 1;
                foreach (var confirm in confirms)
                {
                    resp = DoSubmit(confidence, quality, artist, title, barcode, false, confirm, AccurateRipVerify.maxNpar);
                    if (resp.ParityNeeded)
                        resp = DoSubmit(confidence, quality, artist, title, barcode, true, confirm, Math.Min(AccurateRipVerify.maxNpar, resp.npar));
                    subResult = subResult + (subResult == "" ? "" : ", ") + resp.message;
                }
                return subResult;
            }
            resp = DoSubmit(confidence, quality, artist, title, barcode, false, null, AccurateRipVerify.maxNpar);
			if (resp.ParityNeeded)
				resp = DoSubmit(confidence, quality, artist, title, barcode, true, null, Math.Min(AccurateRipVerify.maxNpar, resp.npar));
            subResult = resp.message;
            return subResult;
		}

        protected CTDBSubmitResponse DoSubmit(int confidence, int quality, string artist, string title, string barcode, bool upload, DBEntry confirm, int npar)
        {
            var files = new List<UploadFile>();
            long maxId = 0;
            foreach (var e in this.entries)
            {
                maxId = Math.Max(maxId, e.id);
            }

            HttpWebRequest req = (HttpWebRequest)WebRequest.Create(urlbase + "/submit2.php");
            req.Proxy = proxy;
            req.UserAgent = this.userAgent;
            req.Timeout = connectTimeout;
            req.ReadWriteTimeout = socketTimeout;
            NameValueCollection form = new NameValueCollection();
            int offset = 0;
            if (confirm != null)
            {
                offset = -confirm.offset;

                // Optional sanity check: should be done by server
                
                if (verify.AR.CTDBCRC(offset) != confirm.crc)
                    throw new Exception("crc mismatch");

                if (confirm.trackcrcs != null)
                {
                    bool crcEquals = true;
                    for (int i = 0; i < confirm.trackcrcs.Length; i++)
                        crcEquals &= verify.TrackCRC(i + 1, offset) == confirm.trackcrcs[i];
                    if (!crcEquals)
                        throw new Exception("track crc mismatch");
                }

                var syn2 = verify.AR.GetSyndrome(confirm.Npar, 1, offset);
                bool equals = true;
                for (int i = 0; i < confirm.Npar; i++)
                    equals &= confirm.syndrome[0, i] == syn2[0, i];
                if (!equals)
                    throw new Exception("syndrome mismatch");
            }
            if (upload)
            {
                files.Add(new UploadFile(new MemoryStream(ParityToSyndrome.Syndrome2Bytes(verify.AR.GetSyndrome(npar, -1, offset))), "parityfile", "data.bin", "application/octet-stream"));
                form.Add("parityfile", "1");
            }
            form.Add("parity", Convert.ToBase64String(ParityToSyndrome.Syndrome2Parity(verify.AR.GetSyndrome(8, 1, offset))));
            form.Add("syndrome", Convert.ToBase64String(ParityToSyndrome.Syndrome2Bytes(verify.AR.GetSyndrome(npar, 1, offset))));
            if (confirm != null)
                form.Add("confirmid", confirm.id.ToString());
            form.Add("ctdb", "2");
            form.Add("npar", npar.ToString());
            form.Add("maxid", maxId.ToString());
            form.Add("toc", toc.ToString());
            form.Add("crc32", ((int)verify.AR.CTDBCRC(offset)).ToString());
            form.Add("trackcrcs", verify.GetTrackCRCs(offset));
            form.Add("confidence", confidence.ToString());
            form.Add("userid", GetUUID());
            form.Add("quality", quality.ToString());
            if (driveName != null && driveName != "") form.Add("drivename", driveName);
            if (barcode != null && barcode != "") form.Add("barcode", barcode);
            if (artist != null && artist != "") form.Add("artist", artist);
            if (title != null && title != "") form.Add("title", title);

            currentReq = req;
            try
            {
                using (HttpWebResponse resp = uploadHelper.Upload(req, files.ToArray(), form))
                {
                    if (resp.StatusCode == HttpStatusCode.OK)
                    {
                        using (Stream s = resp.GetResponseStream())
                        {
                            var serializer = new XmlSerializer(typeof(CTDBSubmitResponse));
                            return serializer.Deserialize(s) as CTDBSubmitResponse;
                        }
                    }
                    else
                    {
                        return new CTDBSubmitResponse() { status = "database access error", message = resp.StatusCode.ToString() };
                    }
                }
            }
            catch (WebException ex)
            {
                return new CTDBSubmitResponse() { status = "database access error", message = ex.Message ?? ex.Status.ToString() };
            }
            catch (Exception ex)
            {
                return new CTDBSubmitResponse() { status = "database access error", message = ex.Message };
            }
            finally
            {
                currentReq = null;
            }
        }

		public void DoVerify()
		{
			if (this.QueryExceptionStatus != WebExceptionStatus.Success)
				return;
			foreach (DBEntry entry in entries)
			{
                if (entry.toc.AudioLength - entry.toc.Pregap != toc.AudioLength - toc.Pregap || entry.stride != verify.Stride)
				{
					entry.hasErrors = true;
					entry.canRecover = false;
					continue;
				}
				if (!verify.FindOffset(entry.syndrome, entry.crc, out entry.offset, out entry.hasErrors))
					entry.canRecover = false;
				else if (entry.hasErrors)
				{
					if (entry.hasParity == null || entry.hasParity == "")
						entry.canRecover = false;
					else
					{
                        ushort[,] syn = null;
                        for (int npar = 4; npar <= Math.Min(entry.Npar, AccurateRipVerify.maxNpar); npar *= 2)
                        {
                            syn = FetchDB(entry, npar, syn);
                            if (entry.httpStatus != HttpStatusCode.OK)
                            {
                                entry.canRecover = false;
                                break;
                            }
                            npar = syn.GetLength(1);
                            entry.repair = verify.VerifyParity(syn, entry.crc, entry.offset);
                            entry.canRecover = entry.repair.CanRecover;
                            if (entry.canRecover)
                            {
                                // entry.syndrome = syn;
                                break;
                            }
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

		public List<DBEntry> MatchingEntries
		{
            get
            {
                var res = new List<DBEntry>();
                if (this.QueryExceptionStatus != WebExceptionStatus.Success)
                    return res;
                foreach (DBEntry entry in this.Entries)
                    if (entry.toc.ToString() == this.toc.ToString() && !entry.hasErrors)
                        res.Add(entry);
                return res;
            }
		}

		public void Init(AccurateRipVerify ar)
		{
			verify = new CDRepairEncode(ar, 10 * 588 * 2);
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

        public void GenerateLog(TextWriter sw, bool old)
        {
            if (this.DBStatus != null || this.Total == 0)
                return;

            if (old)
            {
                sw.WriteLine("        [ CTDBID ] Status");
                foreach (DBEntry entry in this.Entries)
                {
                    string confFormat = (this.Total < 10) ? "{0:0}/{1:0}" :
                        (this.Total < 100) ? "{0:00}/{1:00}" : "{0:000}/{1:000}";
                    string conf = string.Format(confFormat, entry.conf, this.Total);
                    string dataTrackInfo = !entry.toc[entry.toc.TrackCount].IsAudio && this.toc[entry.toc.TrackCount].IsAudio ?
                        string.Format("CD-Extra data track length {0}", entry.toc[entry.toc.TrackCount].LengthMSF) :
                        !entry.toc[1].IsAudio && this.toc[1].IsAudio ?
                        string.Format("Playstation type data track length {0}", entry.toc[entry.toc.FirstAudio].StartMSF) :
                        (entry.toc[1].IsAudio && !this.toc[1].IsAudio) || (entry.toc[entry.toc.TrackCount].IsAudio && !this.toc[entry.toc.TrackCount].IsAudio) ?
                        "Has no data track" : "";
                    if (entry.toc.Pregap != this.toc.Pregap)
                        dataTrackInfo = dataTrackInfo + (dataTrackInfo == "" ? "" : ", ") + string.Format("Has pregap length {0}", CDImageLayout.TimeToString(entry.toc.Pregap));
                    string status =
                        entry.toc.AudioLength - entry.toc.Pregap != this.TOC.AudioLength - this.TOC.Pregap ? string.Format("Has audio length {0}", CDImageLayout.TimeToString(entry.toc.AudioLength)) :
                        ((entry.toc.TrackOffsets != this.TOC.TrackOffsets) ? dataTrackInfo + ", " : "") +
                            ((!entry.hasErrors) ? "Accurately ripped" :
                        //((!entry.hasErrors) ? string.Format("Accurately ripped, offset {0}", -entry.offset) :
                            entry.canRecover ? string.Format("Differs in {0} samples @{1}", entry.repair.CorrectableErrors, entry.repair.AffectedSectors) :
                            (entry.httpStatus == 0 || entry.httpStatus == HttpStatusCode.OK) ? "No match" :
                            entry.httpStatus.ToString());
                    sw.WriteLine("        [{0:x8}] ({1}) {2}", entry.crc, conf, status);
                }
            }

            const int _arOffsetRange = 5 * 588 - 1;
            sw.WriteLine("Track | CTDB Status");
            string ifmt = this.Total < 10 ? "1" : this.Total < 100 ? "2" : "3";
            for (int iTrack = 0; iTrack < this.TOC.AudioTracks; iTrack++)
            {
                int conf = 0;
                List<int> resConfidence = new List<int>();
                List<string> resStatus = new List<string>();
                foreach (DBEntry entry in this.Entries)
                {
                    if (!entry.hasErrors)
                    {
                        conf += entry.conf;
                        continue;
                    }
                    if (entry.canRecover)
                    {
                        var tri = this.TOC[this.TOC.FirstAudio + iTrack];
                        var tr0 = this.TOC[this.TOC.FirstAudio];
                        var min = (int)(tri.Start - tr0.Start) * 588;
                        var max = (int)(tri.End + 1 - tr0.Start) * 588;
                        var diffCount = entry.repair.GetAffectedSectorsCount(min, max);
                        if (diffCount == 0)
                        {
                            conf += entry.conf;
                            continue;
                        }

                        resConfidence.Add(entry.conf);
                        resStatus.Add(string.Format("differs in {0} samples @{1}", diffCount, entry.repair.GetAffectedSectors(min, max)));
                        continue;
                    }
                    if (entry.trackcrcs != null)
                    {
                        if (this.verify.TrackCRC(iTrack + 1, -entry.offset) == entry.trackcrcs[iTrack])
                        {
                            conf += entry.conf;
                            continue;
                        }
                        for (int oi = -_arOffsetRange; oi <= _arOffsetRange; oi++)
                        {
                            if (this.verify.TrackCRC(iTrack + 1, oi) == entry.trackcrcs[iTrack])
                            {
                                conf += entry.conf;
                                break;
                            }
                        }
                    }
                }
                if (conf > 0)
                {
                    resConfidence.Insert(0, conf);
                    resStatus.Insert(0, "accurately ripped");
                }
                if (resStatus.Count == 0)
                {
                    resConfidence.Add(0);
                    resStatus.Add("no match");
                }
                resStatus[0] = string.Format("({0," + ifmt + "}/{1}) {2}", resConfidence[0], this.Total, char.ToUpper(resStatus[0][0]) + resStatus[0].Substring(1));
                for (int i = 1; i < resStatus.Count; i++)
                {
                    resStatus[i] = string.Format("({0}/{1}) {2}", resConfidence[i], this.Total, resStatus[i]);
                }
                sw.WriteLine(string.Format(" {0,2}   | {1}", iTrack + 1, string.Join(", or ", resStatus.ToArray())));
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
