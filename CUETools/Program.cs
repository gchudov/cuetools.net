using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using CUETools.Processor;
using CUETools.Processor.Settings;

namespace JDP
{
	static class Program {
		[STAThread]
		static void Main(string[] args)
		{
			if (args.Length > 1 && args[0].Length > 1 && args[0][0] == '/')
			{
				Application.EnableVisualStyles();
				Application.SetCompatibleTextRenderingDefault(false);
				frmBatch batch = new frmBatch();
				batch.Profile = args[0].Substring(1);

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
			config.Load(new SettingsReader("CUE Tools", "settings.txt", Application.ExecutablePath));
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

    class SingletonController
    {
		private static Mutex m_Mutex = null;
		private static string m_PipeName = null;
		private static Thread m_ListenerThread = null;
		private static CancellationTokenSource m_Cancellation = null;

		public delegate bool ReceiveDelegate(string[] args);

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
			bool createdNew;
			try
			{
				m_Mutex = new Mutex(true, "CUETools-" + id, out createdNew);
			}
			catch
			{
				return false;
			}

			if (!createdNew)
			{
				m_Mutex.Dispose();
				m_Mutex = null;
				return false;
			}

			m_PipeName = "CUETools-" + id;
			m_Cancellation = new CancellationTokenSource();
			m_ListenerThread = new Thread(Listen);
			m_ListenerThread.IsBackground = true;
			m_ListenerThread.Name = "CUETools single-instance listener";
			m_ListenerThread.Start();
			return true;
		}

        public static void Cleanup()
        {
			CancellationTokenSource cancellation = m_Cancellation;
			string pipeName = m_PipeName;
			Thread listenerThread = m_ListenerThread;

			if (cancellation != null)
			{
				cancellation.Cancel();
				try
				{
					using (NamedPipeClientStream client = new NamedPipeClientStream(".", pipeName, PipeDirection.Out))
						client.Connect(50);
				}
				catch
				{
				}
			}

			if (listenerThread != null)
				listenerThread.Join(1000);
			m_ListenerThread = null;
			m_Cancellation = null;
			m_PipeName = null;

			if (m_Mutex != null)
			{
				try
				{
					m_Mutex.ReleaseMutex();
				}
				catch
				{
				}
				m_Mutex.Dispose();
			}
			m_Mutex = null;
        }

        public static void Send(string id, string[] s)
        {
			bool result = false;
            try
            {
				using (NamedPipeClientStream client = new NamedPipeClientStream(".", "CUETools-" + id, PipeDirection.InOut))
				{
					client.Connect(1000);
					BinaryWriter writer = new BinaryWriter(client, Encoding.UTF8, true);
					writer.Write(s.Length);
					foreach (string arg in s)
						writer.Write(arg ?? "");
					writer.Flush();

					BinaryReader reader = new BinaryReader(client, Encoding.UTF8, true);
					result = reader.ReadBoolean();
				}
			}
            catch
            {
            }
			if (!result)
				MessageBox.Show("Another instance of the application seems to be running, but not responding.",
					"Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
		}

		private static void Listen()
		{
			while (true)
			{
				CancellationTokenSource cancellation = m_Cancellation;
				string pipeName = m_PipeName;
				if (cancellation == null || cancellation.IsCancellationRequested || string.IsNullOrEmpty(pipeName))
					return;

				try
				{
					using (NamedPipeServerStream server = new NamedPipeServerStream(pipeName, PipeDirection.InOut, 1))
					{
						server.WaitForConnection();
						if (cancellation.IsCancellationRequested)
							continue;

						BinaryReader reader = new BinaryReader(server, Encoding.UTF8, true);
						int count = reader.ReadInt32();
						string[] args = new string[count];
						for (int i = 0; i < count; i++)
							args[i] = reader.ReadString();

						bool result = Receive(args);
						BinaryWriter writer = new BinaryWriter(server, Encoding.UTF8, true);
						writer.Write(result);
						writer.Flush();
					}
				}
				catch
				{
				}
			}
		}

        private static bool Receive(string[] s)
        {
			if (m_Receive == null)
				return false;
             return m_Receive(s);
        }
    }
}
