using System;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Threading;
using System.Windows.Forms;
using CUETools.Processor;
using CUETools.Processor.Settings;

namespace CUERipper
{
	static class Program
	{
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main()
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);

			string arch = Marshal.SizeOf(typeof(IntPtr)) == 8 ? "x64" : "win32";
			GetSatelliteAssemblies(System.IO.Path.Combine("plugins", arch));

			CUEConfig config = new CUEConfig();
			config.Load(new SettingsReader("CUERipper", "settings.txt", Application.ExecutablePath));
			try { Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(config.language); }
			catch { }

			Application.Run(new frmCUERipper());
		}

		static void GetSatelliteAssemblies(string groupName)
		{
			// System.Deployment.Application is not available on .NET 10.
			// Optional plugin file groups must be included by the publish/package step.
		}
	}
}
