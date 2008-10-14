using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace JDP {
	public partial class frmSettings : Form {
		int _writeOffset;
		CUEConfig _config;

		public frmSettings() {
			InitializeComponent();
		}

		private void frmSettings_Load(object sender, EventArgs e) {
			numericWriteOffset.Value = _writeOffset;
			chkPreserveHTOA.Checked = _config.preserveHTOA;
			chkAutoCorrectFilenames.Checked = _config.autoCorrectFilenames;
			numericFLACCompressionLevel.Value = _config.flacCompressionLevel;
			numFixWhenConfidence.Value = _config.fixWhenConfidence;
			numFixWhenPercent.Value = _config.fixWhenPercent;
			numEncodeWhenConfidence.Value = _config.encodeWhenConfidence;
			numEncodeWhenPercent.Value = _config.encodeWhenPercent;
			chkFLACVerify.Checked = _config.flacVerify;
			chkArAddCRCs.Checked = _config.writeArTags;
			if (_config.wvCompressionMode == 0) rbWVFast.Checked = true;
			if (_config.wvCompressionMode == 1) rbWVNormal.Checked = true;
			if (_config.wvCompressionMode == 2) rbWVHigh.Checked = true;
			if (_config.wvCompressionMode == 3) rbWVVeryHigh.Checked = true;
			chkWVExtraMode.Checked = (_config.wvExtraMode != 0);
			chkWVExtraMode_CheckedChanged(null, null);
			if (_config.wvExtraMode != 0) numWVExtraMode.Value = _config.wvExtraMode;
			chkKeepOriginalFilenames.Checked = _config.keepOriginalFilenames;
			txtSingleFilenameFormat.Text = _config.singleFilenameFormat;
			txtTrackFilenameFormat.Text = _config.trackFilenameFormat;
			chkRemoveSpecial.Checked = _config.removeSpecial;
			txtSpecialExceptions.Text = _config.specialExceptions;
			chkReplaceSpaces.Checked = _config.replaceSpaces;
			chkArSaveLog.Checked = _config.writeArLog;
			chkArNoUnverifiedAudio.Checked = _config.noUnverifiedOutput;
			chkArFixOffset.Checked = _config.fixOffset;
			chkEmbedLog.Checked = _config.embedLog;
		}

		private void frmSettings_FormClosing(object sender, FormClosingEventArgs e) {
		}

		public int WriteOffset {
			get { return _writeOffset; }
			set { _writeOffset = value; }
		}

		public CUEConfig Config {
			get { return _config; }
			set { _config = value; }
		}

		private void chkWVExtraMode_CheckedChanged(object sender, EventArgs e) {
			if (chkWVExtraMode.Checked) {
				numWVExtraMode.Enabled = true;
			}
			else {
				numWVExtraMode.Enabled = false;
			}
		}

		private void btnOK_Click(object sender, EventArgs e)
		{
			_writeOffset = (int)numericWriteOffset.Value;
			_config.preserveHTOA = chkPreserveHTOA.Checked;
			_config.autoCorrectFilenames = chkAutoCorrectFilenames.Checked;
			_config.flacCompressionLevel = (uint)numericFLACCompressionLevel.Value;
			_config.fixWhenPercent = (uint)numFixWhenPercent.Value;
			_config.fixWhenConfidence = (uint)numFixWhenConfidence.Value;
			_config.encodeWhenPercent = (uint)numEncodeWhenPercent.Value;
			_config.encodeWhenConfidence = (uint)numEncodeWhenConfidence.Value;
			_config.flacVerify = chkFLACVerify.Checked;
			_config.writeArTags = chkArAddCRCs.Checked;
			if (rbWVFast.Checked) _config.wvCompressionMode = 0;
			else if (rbWVHigh.Checked) _config.wvCompressionMode = 2;
			else if (rbWVVeryHigh.Checked) _config.wvCompressionMode = 3;
			else _config.wvCompressionMode = 1;
			if (!chkWVExtraMode.Checked) _config.wvExtraMode = 0;
			else _config.wvExtraMode = (int) numWVExtraMode.Value;
			_config.keepOriginalFilenames = chkKeepOriginalFilenames.Checked;
			_config.singleFilenameFormat = txtSingleFilenameFormat.Text;
			_config.trackFilenameFormat = txtTrackFilenameFormat.Text;
			_config.removeSpecial = chkRemoveSpecial.Checked;
			_config.specialExceptions = txtSpecialExceptions.Text;
			_config.replaceSpaces = chkReplaceSpaces.Checked;
			_config.writeArLog = chkArSaveLog.Checked;
			_config.noUnverifiedOutput = chkArNoUnverifiedAudio.Checked;
			_config.fixOffset = chkArFixOffset.Checked;
			_config.embedLog = chkEmbedLog.Checked;
		}

		private void chkArFixOffset_CheckedChanged(object sender, EventArgs e)
		{
			numFixWhenConfidence.Enabled = chkArFixOffset.Checked;
			numFixWhenPercent.Enabled = chkArFixOffset.Checked;
		}

		private void chkArNoUnverifiedAudio_CheckedChanged(object sender, EventArgs e)
		{
			numEncodeWhenConfidence.Enabled = chkArNoUnverifiedAudio.Checked;
			numEncodeWhenPercent.Enabled = chkArNoUnverifiedAudio.Checked;
		}
	}
}