using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace JDP {
	static class Program {
		[STAThread]
		static void Main(string[] args)
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			if (args.Length == 2 && (args[0] == "/verify" || args[0] == "/convert" || args[0] == "/fix"))
			{
				frmBatch batch = new frmBatch();
				batch.InputPath = args[1];
				batch.AccurateRip = (args[0] != "/convert");
				batch.AccurateOffset = (args[0] == "/fix");
				Application.Run (batch);
				return;
			}
			frmCUETools form = new frmCUETools();
			if (args.Length == 1)
				form.InputPath = args[0];
			Application.Run(form);
		}
	}
}