using System;
using System.Deployment.Application;
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
			if (ApplicationDeployment.IsNetworkDeployed)
			{
				ApplicationDeployment deploy = ApplicationDeployment.CurrentDeployment;

				if (deploy.IsFirstRun)
				{
					try
					{
						deploy.DownloadFileGroup(groupName);
					}
					catch (DeploymentException)
					{
						// Log error. Do not report this error to the user, because a satellite
						// assembly may not exist if the user's culture and the application's
						// default culture match.
					}
				}
			}
		}
	}
}