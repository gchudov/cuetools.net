using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Threading;

namespace JDP {
	public partial class frmFilenameCorrector : Form {
		private Thread _workThread;

		public frmFilenameCorrector() {
			InitializeComponent();
		}

		private void frmFilenameCorrector_DragEnter(object sender, DragEventArgs e) {
			if (e.Data.GetDataPresent(DataFormats.FileDrop)) {
				e.Effect = DragDropEffects.Copy;
			}
		}

		private void frmFilenameCorrector_DragDrop(object sender, DragEventArgs e) {
			if ((_workThread != null) && (_workThread.IsAlive)) {
				return;
			}
			if (e.Data.GetDataPresent(DataFormats.FileDrop)) {
				_workThread = new Thread(new ParameterizedThreadStart(CorrectCUEThread));
				_workThread.Start(e.Data.GetData(DataFormats.FileDrop));
			}
		}

		private void CorrectCUEThread(object p) {
			string[] paths = (string[])p;
			bool oneSuccess = false;
			bool cancel = false;

			foreach (string path in paths) {
				if (Path.GetExtension(path).ToLower() == ".cue") {
					try {
						string fixedCue = CUESheet.CorrectAudioFilenames(path, true);
						using (StreamWriter sw = new StreamWriter(path, false, CUESheet.Encoding))
							sw.Write (fixedCue);
						oneSuccess = true;
					}
					catch (Exception ex) {
						Invoke((MethodInvoker)delegate() {
							cancel = (MessageBox.Show(this, path + Environment.NewLine +
								Environment.NewLine + ex.Message, "Error", MessageBoxButtons.OKCancel,
								MessageBoxIcon.Error) == DialogResult.Cancel);
						});
						if (cancel) break;
					}
				}
			}

			if (oneSuccess) {
				Invoke((MethodInvoker)delegate() {
					MessageBox.Show(this, "Filename correction is complete!", "Done",
						MessageBoxButtons.OK, MessageBoxIcon.Information);
				});
			}
		}
	}
}