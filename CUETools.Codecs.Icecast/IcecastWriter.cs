using System;
using System.Collections.Generic;
using System.Text;
using System.Net;
using System.IO;
using System.Web;
using CUETools.Codecs;

namespace CUETools.Codecs.Icecast
{
	public class IcecastWriter: IAudioDest
	{
		private long _sampleOffset = 0;
        private AudioEncoderSettings m_settings;
		private LAME.LAMEEncoderCBR encoder = null;
		private HttpWebRequest req = null;
		private HttpWebResponse resp = null;
		private Stream reqStream;
		private IcecastSettingsData settings = null;

		public IAudioDest Encoder
		{
			get
			{
				return encoder;
			}
		}

		public IcecastWriter(AudioPCMConfig pcm, IcecastSettingsData settings)
		{
            this.m_settings = new AudioEncoderSettings(pcm);
			this.settings = settings;
		}

		#region IAudioDest Members

		public HttpWebResponse Response
		{
			get
			{
				return resp;
			}
		}

		public void Connect()
		{
			Uri uri = new Uri("http://" + settings.Server + ":" + settings.Port + settings.Mount);
			req = (HttpWebRequest)WebRequest.Create(uri);
			//req.Proxy = proxy;
			//req.UserAgent = userAgent;
			req.ProtocolVersion = HttpVersion.Version10; // new Version("ICE/1.0");
			req.Method = "SOURCE";
			req.ContentType = "audio/mpeg";
			req.Headers.Add("ice-name", settings.Name ?? "no name");
			req.Headers.Add("ice-public", "1");
			if ((settings.Url ?? "") != "") req.Headers.Add("ice-url", settings.Url);
			if ((settings.Genre ?? "") != "") req.Headers.Add("ice-genre", settings.Genre);
			if ((settings.Desctiption ?? "") != "") req.Headers.Add("ice-description", settings.Desctiption);
			req.Headers.Add("Authorization", string.Format("Basic {0}", Convert.ToBase64String(Encoding.ASCII.GetBytes(string.Format("source:{0}", settings.Password)))));
			req.Timeout = System.Threading.Timeout.Infinite;
			req.ReadWriteTimeout = System.Threading.Timeout.Infinite;
			//req.ContentLength = 999999999;
			req.KeepAlive = false;
			req.SendChunked = true;
			req.AllowWriteStreamBuffering = false;
			req.CachePolicy = new System.Net.Cache.HttpRequestCachePolicy(System.Net.Cache.HttpRequestCacheLevel.BypassCache);

			System.Reflection.PropertyInfo pi = typeof(ServicePoint).GetProperty("HttpBehaviour", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
			pi.SetValue(req.ServicePoint, pi.PropertyType.GetField("Unknown").GetValue(null), null);

			reqStream = req.GetRequestStream();

			System.Reflection.FieldInfo fi = reqStream.GetType().GetField("m_HttpWriteMode", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
			fi.SetValue(reqStream, fi.FieldType.GetField("Buffer").GetValue(null));
			System.Reflection.MethodInfo mi = reqStream.GetType().GetMethod("CallDone", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic, null, new Type[0], null);
			mi.Invoke(reqStream, null);

			try
			{
				resp = req.GetResponse() as HttpWebResponse;
				if (resp.StatusCode == HttpStatusCode.OK)
				{
                    var encoderSettings = new CUETools.Codecs.LAME.LAMEEncoderCBRSettings() { PCM = AudioPCMConfig.RedBook };
                    encoderSettings.StereoMode = settings.JointStereo ?
                        CUETools.Codecs.LAME.Interop.MpegMode.JOINT_STEREO :
                        CUETools.Codecs.LAME.Interop.MpegMode.STEREO;
                    encoderSettings.CustomBitrate = settings.Bitrate;
                    encoder = new CUETools.Codecs.LAME.LAMEEncoderCBR("", reqStream, encoderSettings);
				}
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					resp = ex.Response as HttpWebResponse;
				else
					throw ex;
			}
		}

		public void UpdateMetadata(string artist, string title)
		{
			string song = ((artist ?? "") != "" && (title ?? "") != "") ? artist + " - " + title : (title ?? "");
			string metadata = "";
			//if (station != "")
			//    metadata += "&name=" + Uri.EscapeDataString(station);
			if (song != "")
				metadata += "&song=" + Uri.EscapeDataString(song);
			Uri uri = new Uri("http://" + settings.Server + ":" + settings.Port + "/admin/metadata?mode=updinfo&mount=" + settings.Mount + metadata);
			HttpWebRequest req2 = (HttpWebRequest)WebRequest.Create(uri);
			req2.Method = "GET";
			req2.Credentials = new NetworkCredential("source", settings.Password);
			//req.Proxy = proxy;
			//req.UserAgent = userAgent;
			//req2.Headers.Add("Authorization", string.Format("Basic {0}", Convert.ToBase64String(Encoding.ASCII.GetBytes(string.Format("source:{0}", settings.Password)))));
			HttpStatusCode accResult = HttpStatusCode.OK;
			try
			{
				HttpWebResponse resp = (HttpWebResponse)req2.GetResponse();
				accResult = resp.StatusCode;
				if (accResult == HttpStatusCode.OK)
				{
				}
				resp.Close();
			}
			catch (WebException ex)
			{
				if (ex.Status == WebExceptionStatus.ProtocolError)
					accResult = ((HttpWebResponse)ex.Response).StatusCode;
				else
					accResult = HttpStatusCode.BadRequest;
			}
		}

		public void Close()
		{
			if (encoder != null)
			{
				encoder.Close();
				encoder = null;
			}
			if (reqStream != null)
			{
				reqStream.Close();
				reqStream = null;
			}
			if (resp != null)
			{
				resp.Close();
				resp = null;
			}
			if (req != null)
			{
				req.Abort();
				req = null;
			}
		}

		public void Delete()
		{
			if (encoder != null)
			{
				encoder.Delete();
				encoder = null;
			}
			if (reqStream != null)
			{
				reqStream.Close();
				reqStream = null;
			}
			if (resp != null)
			{
				resp.Close();
				resp = null;
			}
			if (req != null)
			{
				req.Abort();
				req = null;
			}
		}

		AudioBuffer tmp;

		public void Write(AudioBuffer src)
		{
			if (encoder == null)
				throw new Exception("not connected");

			if (tmp == null || tmp.Size < src.Size)
				tmp = new AudioBuffer(AudioPCMConfig.RedBook, src.Size);
			tmp.Prepare(-1);
			Buffer.BlockCopy(src.Float, 0, tmp.Float, 0, src.Length * 8);
			tmp.Length = src.Length;
			encoder.Write(tmp);
		}

		public long Position
		{
			get
			{
				return _sampleOffset;
			}
		}

		public long FinalSampleCount
		{
			set { ; }
		}

        public AudioEncoderSettings Settings
		{
			get
			{
				return m_settings;
			}
		}

		public string Path { get { return null; } }
		#endregion

		public long BytesWritten
		{
			get
			{
				return encoder == null ? 0 : encoder.BytesWritten;
			}
		}
	}

	public class IcecastSettingsData
	{
		public IcecastSettingsData()
		{
			Port = "8000";
			Bitrate = 192;
			JointStereo = true;
		}

		private string server;
		private string password;
		private string mount;
		private string name;
		private string description;
		private string url;
		private string genre;

		public string Server { get { return server; } set { server = value; } }
		public string Port { get; set; }
		public string Password { get { return password; } set { password = value; } }
		public string Mount { get { return mount; } set { mount = value; } }
		public string Name { get { return name; } set { name = value; } }
		public string Desctiption { get { return description; } set { description = value; } }
		public string Url { get { return url; } set { url = value; } }
		public string Genre { get { return genre; } set { genre = value; } }
		public int    Bitrate { get; set; }
		public bool   JointStereo { get; set; }
	}
}
