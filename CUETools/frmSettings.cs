using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using CUETools.Processor;

namespace JDP {
	public partial class frmSettings : Form {
		int _writeOffset;
		bool _reducePriority;
		CUEConfig _config;

		public frmSettings() {
			InitializeComponent();
		}

		private void frmSettings_Load(object sender, EventArgs e) {
			chkReducePriority.Checked = _reducePriority;
			numericWriteOffset.Value = _writeOffset;
			chkPreserveHTOA.Checked = _config.preserveHTOA;
			chkAutoCorrectFilenames.Checked = _config.autoCorrectFilenames;
			numericFLACCompressionLevel.Value = _config.flacCompressionLevel;
			numFixWhenConfidence.Value = _config.fixWhenConfidence;
			numFixWhenPercent.Value = _config.fixWhenPercent;
			numEncodeWhenConfidence.Value = _config.encodeWhenConfidence;
			numEncodeWhenPercent.Value = _config.encodeWhenPercent;
			chkEncodeWhenZeroOffset.Checked = _config.encodeWhenZeroOffset;
			chkFLACVerify.Checked = _config.flacVerify;
			chkWriteArTagsOnConvert.Checked = _config.writeArTagsOnConvert;
			chkWriteARTagsOnVerify.Checked = _config.writeArTagsOnVerify;
			if (_config.wvCompressionMode == 0) rbWVFast.Checked = true;
			if (_config.wvCompressionMode == 1) rbWVNormal.Checked = true;
			if (_config.wvCompressionMode == 2) rbWVHigh.Checked = true;
			if (_config.wvCompressionMode == 3) rbWVVeryHigh.Checked = true;
			chkWVExtraMode.Checked = (_config.wvExtraMode != 0);
			if (_config.wvExtraMode != 0) numWVExtraMode.Value = _config.wvExtraMode;
			chkWVStoreMD5.Checked = _config.wvStoreMD5;
			switch (_config.apeCompressionLevel)
			{
				case 1: rbAPEfast.Checked = true; break;
				case 2: rbAPEnormal.Checked = true; break;
				case 3: rbAPEhigh.Checked = true; break;
				case 4: rbAPEextrahigh.Checked = true; break;
				case 5: rbAPEinsane.Checked = true; break;
			}
			chkKeepOriginalFilenames.Checked = _config.keepOriginalFilenames;
			txtSingleFilenameFormat.Text = _config.singleFilenameFormat;
			txtTrackFilenameFormat.Text = _config.trackFilenameFormat;
			chkRemoveSpecial.Checked = _config.removeSpecial;
			txtSpecialExceptions.Text = _config.specialExceptions;
			chkReplaceSpaces.Checked = _config.replaceSpaces;
			chkWriteArLogOnConvert.Checked = _config.writeArLogOnConvert;
			chkWriteARLogOnVerify.Checked = _config.writeArLogOnVerify;
			chkArNoUnverifiedAudio.Checked = _config.noUnverifiedOutput;
			chkArFixOffset.Checked = _config.fixOffset;
			chkEmbedLog.Checked = _config.embedLog;
			chkFillUpCUE.Checked = _config.fillUpCUE;
			chkFilenamesANSISafe.Checked = _config.filenamesANSISafe;
			chkHDCDDetect.Checked = _config.detectHDCD;
			chkHDCDDecode.Checked = _config.decodeHDCD;
			chkHDCDStopLooking.Checked = _config.wait750FramesForHDCD;
			chkCreateM3U.Checked = _config.createM3U;
			chkCreateCUEFileWhenEmbedded.Checked = _config.createCUEFileWhenEmbedded;
			chkTruncateExtra4206Samples.Checked = _config.truncate4608ExtraSamples;
			numericLossyWAVQuality.Value = _config.lossyWAVQuality;
			chkHDCDLW16.Checked = _config.decodeHDCDtoLW16;
			chkHDCD24bit.Checked = _config.decodeHDCDto24bit;

			EnableDisable();
		}

		private void frmSettings_FormClosing(object sender, FormClosingEventArgs e) {
		}

		public int WriteOffset {
			get { return _writeOffset; }
			set { _writeOffset = value; }
		}

		public bool ReducePriority
		{
			get { return _reducePriority; }
			set { _reducePriority = value; }
		}

		public CUEConfig Config {
			get { return _config; }
			set { _config = value; }
		}

		private void chkWVExtraMode_CheckedChanged(object sender, EventArgs e) {
			EnableDisable();
		}

		private void btnOK_Click(object sender, EventArgs e)
		{
			_writeOffset = (int)numericWriteOffset.Value;
			_reducePriority = chkReducePriority.Checked;
			_config.preserveHTOA = chkPreserveHTOA.Checked;
			_config.autoCorrectFilenames = chkAutoCorrectFilenames.Checked;
			_config.flacCompressionLevel = (uint)numericFLACCompressionLevel.Value;
			_config.lossyWAVQuality = (int)numericLossyWAVQuality.Value;
			_config.fixWhenPercent = (uint)numFixWhenPercent.Value;
			_config.fixWhenConfidence = (uint)numFixWhenConfidence.Value;
			_config.encodeWhenPercent = (uint)numEncodeWhenPercent.Value;
			_config.encodeWhenConfidence = (uint)numEncodeWhenConfidence.Value;
			_config.encodeWhenZeroOffset = chkEncodeWhenZeroOffset.Checked;
			_config.flacVerify = chkFLACVerify.Checked;
			_config.writeArTagsOnConvert = chkWriteArTagsOnConvert.Checked;
			_config.writeArTagsOnVerify = chkWriteARTagsOnVerify.Checked;
			if (rbWVFast.Checked) _config.wvCompressionMode = 0;
			else if (rbWVHigh.Checked) _config.wvCompressionMode = 2;
			else if (rbWVVeryHigh.Checked) _config.wvCompressionMode = 3;
			else _config.wvCompressionMode = 1;
			if (!chkWVExtraMode.Checked) _config.wvExtraMode = 0;
			else _config.wvExtraMode = (int) numWVExtraMode.Value;
			_config.wvStoreMD5 = chkWVStoreMD5.Checked;
			_config.apeCompressionLevel = (uint) (rbAPEfast.Checked ? 1 :
				rbAPEnormal.Checked ? 2 :
				rbAPEhigh.Checked ? 3 :
				rbAPEextrahigh.Checked ? 4 :
				rbAPEinsane.Checked ? 5 : 2);
			_config.keepOriginalFilenames = chkKeepOriginalFilenames.Checked;
			_config.singleFilenameFormat = txtSingleFilenameFormat.Text;
			_config.trackFilenameFormat = txtTrackFilenameFormat.Text;
			_config.removeSpecial = chkRemoveSpecial.Checked;
			_config.specialExceptions = txtSpecialExceptions.Text;
			_config.replaceSpaces = chkReplaceSpaces.Checked;
			_config.writeArLogOnConvert = chkWriteArLogOnConvert.Checked;
			_config.writeArLogOnVerify = chkWriteARLogOnVerify.Checked;
			_config.noUnverifiedOutput = chkArNoUnverifiedAudio.Checked;
			_config.fixOffset = chkArFixOffset.Checked;
			_config.embedLog = chkEmbedLog.Checked;
			_config.fillUpCUE = chkFillUpCUE.Checked;
			_config.filenamesANSISafe = chkFilenamesANSISafe.Checked;
			_config.detectHDCD = chkHDCDDetect.Checked;
			_config.wait750FramesForHDCD = chkHDCDStopLooking.Checked;
			_config.decodeHDCD = chkHDCDDecode.Checked;
			_config.createM3U = chkCreateM3U.Checked;
			_config.createCUEFileWhenEmbedded = chkCreateCUEFileWhenEmbedded.Checked;
			_config.truncate4608ExtraSamples = chkTruncateExtra4206Samples.Checked;
			_config.decodeHDCDtoLW16 = chkHDCDLW16.Checked;
			_config.decodeHDCDto24bit = chkHDCD24bit.Checked;
		}

		private void EnableDisable()
		{
			numFixWhenConfidence.Enabled =
			labelFixWhenConfidence.Enabled =
			numFixWhenPercent.Enabled =
			labelFixWhenPercent.Enabled = chkArFixOffset.Checked;

			numEncodeWhenConfidence.Enabled =
			labelEncodeWhenConfidence.Enabled =
			numEncodeWhenPercent.Enabled =
			labelEncodeWhenPercent.Enabled =
			chkEncodeWhenZeroOffset.Enabled = chkArNoUnverifiedAudio.Checked;

			grpHDCD.Enabled = chkHDCDDetect.Checked;
			chkHDCDLW16.Enabled = chkHDCDDetect.Checked && chkHDCDDecode.Checked;
			chkHDCD24bit.Enabled = chkHDCDDetect.Checked && chkHDCDDecode.Checked;

			chkRemoveSpecial.Enabled = chkFilenamesANSISafe.Checked;
			txtSpecialExceptions.Enabled = chkRemoveSpecial.Checked && chkFilenamesANSISafe.Checked;

			txtSpecialExceptions.Enabled = chkRemoveSpecial.Checked;

			numWVExtraMode.Enabled = chkWVExtraMode.Checked;
		}

		private void chkArFixOffset_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkArNoUnverifiedAudio_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkHDCDDetect_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkFilenamesANSISafe_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkRemoveSpecial_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}

		private void chkHDCDDecode_CheckedChanged(object sender, EventArgs e)
		{
			EnableDisable();
		}
	}
}