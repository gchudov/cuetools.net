using System;
using System.Collections.Generic;
using System.Windows.Forms;
using System.IO;
using System.Text;

namespace JDP {
	static class Program {
		[STAThread]
		static void Main(string[] args)
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			if (args.Length > 1 && (args[0] == "/verify" || args[0] == "/convert" || args[0] == "/fix"))
			{
				frmBatch batch = new frmBatch();
				batch.AccurateRip = (args[0] != "/convert");
				batch.AccurateOffset = (args[0] == "/fix");
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
			frmCUETools form = new frmCUETools();
			if (args.Length == 1)
				form.InputPath = args[0];
			Application.Run(form);
		}
	}
}