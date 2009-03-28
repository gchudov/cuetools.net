using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;
using System.Text;
using CUETools.Processor;
using System.Threading;
using System.Globalization;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels.Ipc;

namespace JDP {
	static class Program {
		[STAThread]
		static void Main(string[] args)
		{
			if (args.Length > 1 && (args[0] == "/verify" || args[0] == "/crc" || args[0] == "/convert" || args[0] == "/fix"))
			{
				Application.EnableVisualStyles();
				Application.SetCompatibleTextRenderingDefault(false);
				frmBatch batch = new frmBatch();
				batch.AccurateRip =
					args[0] == "/convert" ? CUEAction.VerifyAndConvert :
					args[0] == "/fix" ? CUEAction.VerifyThenConvert :
					args[0] == "/crc" ? CUEAction.VerifyPlusCRCs :
					CUEAction.Verify;

				if (args.Length == 2 && args[1][0] != '@')
					batch.InputPath = args[1];
				else for (int i = 1; i < args.Length; i++)
				{
					if (args[i][0] == '@')
					{
						string lineStr;
						StreamReader sr;
						try
						{
							sr = new StreamReader(args[i].Substring(1), Encoding.Default);
							while ((lineStr = sr.ReadLine()) != null)
								batch.AddInputPath(lineStr);
						}
						catch
						{
							batch.AddInputPath(args[i]);
						}
					} else
						batch.AddInputPath(args[i]);
				}
				Application.Run(batch);
				return;
			}

			string myId = "BZ92759C-63Q7-444e-ADA6-E495634A493D";
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			CUEConfig config = new CUEConfig();
			config.Load(new SettingsReader("CUE Tools", "settings.txt"));
			try { Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(config.language); }
			catch { }
			frmCUETools form = new frmCUETools();
			if (!config.oneInstance || SingletonController.IamFirst(myId, new SingletonController.ReceiveDelegate(form.OnSecondCall)))
			{
				if (args.Length == 1)
					form.InputPath = args[0];
				Application.Run(form);
			}
			else
			{
				List<string> newArgs = new List<string>();
				foreach (string arg in args)
					newArgs.Add(Path.GetFullPath(arg));
				SingletonController.Send(myId, newArgs.ToArray());
			}
			SingletonController.Cleanup();
		}
	}

	[Serializable]
    class SingletonController : MarshalByRefObject
    {
		private static IpcChannel m_IPCChannel = null;

		public delegate void ReceiveDelegate(string[] args);

        static private ReceiveDelegate m_Receive = null;
        static public ReceiveDelegate Receiver
        {
            get
            {
                return m_Receive;
            }
            set
            {
                m_Receive = value;
            }
        }

        public static bool IamFirst(string id, ReceiveDelegate r)
        {
            if (IamFirst(id))
            {
                Receiver += r;
                return true;
            }
            else
            {
                return false;
            }
        }

        public static bool IamFirst(string id)
        {
			try
			{
				m_IPCChannel = new IpcChannel(id);
			}
			catch
			{
				return false;
			}
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(m_IPCChannel, false);
			RemotingConfiguration.RegisterWellKnownServiceType(
				typeof(SingletonController),
				"SingletonController",
				WellKnownObjectMode.SingleCall);
			return true;
		}

        public static void Cleanup()
        {
            if (m_IPCChannel != null)
                m_IPCChannel.StopListening(null);
            m_IPCChannel = null;
        }

        public static void Send(string id, string[] s)
        {
            SingletonController ctrl;
            IpcChannel channel = new IpcChannel();
			System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(channel, false);
            try
            {
				ctrl = (SingletonController)Activator.GetObject(typeof(SingletonController), "ipc://" + id + "/SingletonController");
				ctrl.Receive(s);
			}
            catch
            {
				MessageBox.Show("Another instance of the application seems to be running, but not responding.",
					"Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public void Receive(string[] s)
        {
            if (m_Receive != null)
            {
                m_Receive(s);
            }
        }
    }
}