using System;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace CUETools.CTDB.EACPlugin
{
    internal sealed class BarcodePrompt : Form
    {
        private readonly TextBox barcodeTextBox;

        public BarcodePrompt()
        {
            this.Text = "Barcode";
            this.FormBorderStyle = FormBorderStyle.FixedToolWindow;
            this.StartPosition = FormStartPosition.CenterParent;
            this.ShowInTaskbar = false;
            this.ClientSize = new Size(320, 28);
            barcodeTextBox = new TextBox();
            barcodeTextBox.Dock = DockStyle.Fill;
            this.Controls.Add(barcodeTextBox);
            this.Shown += delegate { barcodeTextBox.Focus(); };
        }

        public string Barcode
        {
            get { return GetBarcode(this.DialogResult, barcodeTextBox.Text); }
        }

        internal static string GetBarcode(DialogResult result, string value)
        {
            return result == DialogResult.OK ? Sanitize(value) : "";
        }

        internal static string Sanitize(string value)
        {
            if (value == null) return "";
            StringBuilder digits = new StringBuilder(value.Length);
            foreach (char c in value)
                if (c >= '0' && c <= '9') digits.Append(c);
            return digits.ToString();
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == Keys.Enter)
            {
                this.DialogResult = DialogResult.OK;
                return true;
            }
            if (keyData == Keys.Escape)
            {
                this.DialogResult = DialogResult.Cancel;
                return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }
    }
}
