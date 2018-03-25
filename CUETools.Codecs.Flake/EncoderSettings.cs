using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Text;

namespace CUETools.Codecs.Flake
{
    [JsonObject(MemberSerialization.OptIn)]
    public class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "flac";

        [Browsable(false)]
        public string Name => "cuetools";

        [Browsable(false)]
        public Type EncoderType => typeof(AudioEncoder);

        [Browsable(false)]
        public bool Lossless => true;

        [Browsable(false)]
        public int Priority => 4;

        [Browsable(false)]
        public string SupportedModes => this.AllowNonSubset || (this.PCM != null && this.PCM.SampleRate > 48000) ? "0 1 2 3 4 5 6 7 8 9 10 11" : "0 1 2 3 4 5 6 7 8";

        [Browsable(false)]
        public string DefaultMode => "5";

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
        public string EncoderMode { get; set; }

        [Browsable(false)]
        public AudioPCMConfig PCM { get; set; }

        [Browsable(false)]
        public int BlockSize { get; set; }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding { get; set; }

        public IAudioEncoderSettings Clone()
        {
            return MemberwiseClone() as IAudioEncoderSettings;
        }
        #endregion

        public EncoderSettings()
        {
            this.Init();
        }

        public bool IsSubset()
        {
            return (BlockSize == 0 || (BlockSize <= 16384 && (PCM.SampleRate > 48000 || BlockSize <= 4608)))
                && (PCM.SampleRate > 48000 || MaxLPCOrder <= 12)
                && MaxPartitionOrder <= 8
                ;
            //The blocksize bits in the frame header must be 0001-1110. The blocksize must be <=16384; if the sample rate is <= 48000Hz, the blocksize must be <=4608.
            //The sample rate bits in the frame header must be 0001-1110.
            //The bits-per-sample bits in the frame header must be 001-111.
            //If the sample rate is <= 48000Hz, the filter order in LPC subframes must be less than or equal to 12, i.e. the subframe type bits in the subframe header may not be 101100-111111.
            //The Rice partition order in a Rice-coded residual section must be less than or equal to 8.
        }

        public void Validate()
        {
            if (this.GetEncoderModeIndex() < 0)
                throw new Exception("unsupported encoder mode");
            this.SetDefaultValuesForMode();
            if (Padding < 0)
                throw new Exception("unsupported padding value " + Padding.ToString());
            if (BlockSize != 0 && (BlockSize < 256 || BlockSize >= FlakeConstants.MAX_BLOCKSIZE))
                throw new Exception("unsupported block size " + BlockSize.ToString());
            if (MinLPCOrder > MaxLPCOrder || MaxLPCOrder > lpc.MAX_LPC_ORDER)
                throw new Exception("invalid MaxLPCOrder " + MaxLPCOrder.ToString());
            if (MinFixedOrder < 0 || MinFixedOrder > 4)
                throw new Exception("invalid MinFixedOrder " + MinFixedOrder.ToString());
            if (MaxFixedOrder < 0 || MaxFixedOrder > 4)
                throw new Exception("invalid MaxFixedOrder " + MaxFixedOrder.ToString());
            if (MinPartitionOrder < 0)
                throw new Exception("invalid MinPartitionOrder " + MinPartitionOrder.ToString());
            if (MinPartitionOrder > MaxPartitionOrder || MaxPartitionOrder > 8)
                throw new Exception("invalid MaxPartitionOrder " + MaxPartitionOrder.ToString());
            if (PredictionType == PredictionType.None)
                throw new Exception("invalid PredictionType " + PredictionType.ToString());
            if (PredictionType != PredictionType.Fixed)
            {
                if (WindowMethod == WindowMethod.Invalid)
                    throw new InvalidOperationException("invalid WindowMethod " + WindowMethod.ToString());
                if (WindowFunctions == WindowFunction.None)
                    throw new InvalidOperationException("invalid WindowFunctions " + WindowFunctions.ToString());
                if (EstimationDepth > 32 || EstimationDepth < 1)
                    throw new InvalidOperationException("invalid EstimationDepth " + EstimationDepth.ToString());
                if (MinPrecisionSearch < 0 || MinPrecisionSearch >= lpc.MAX_LPC_PRECISIONS)
                    throw new Exception("unsupported MinPrecisionSearch value");
                if (MaxPrecisionSearch < 0 || MaxPrecisionSearch >= lpc.MAX_LPC_PRECISIONS)
                    throw new Exception("unsupported MaxPrecisionSearch value");
                if (MaxPrecisionSearch < MinPrecisionSearch)
                    throw new Exception("unsupported MaxPrecisionSearch value");
            }
            if (!AllowNonSubset && !IsSubset())
                throw new Exception("the encoding parameters specified do not conform to the FLAC Subset");
        }

        [DefaultValue(-1)]
        [DefaultValueForMode(2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0)]
        [Browsable(false)]
        [DisplayName("MinFixedOrder")]
        [SRDescription(typeof(Properties.Resources), "MinFixedOrderDescription")]
        public int MinFixedOrder { get; set; }

        [DefaultValue(-1)]
        [DefaultValueForMode(2, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4)]
        [Browsable(false)]
        [DisplayName("MaxFixedOrder")]
        [SRDescription(typeof(Properties.Resources), "MaxFixedOrderDescription")]
        public int MaxFixedOrder { get; set; }

        [DefaultValue(1)]
        [Browsable(false)]
        [DisplayName("MinLPCOrder")]
        [SRDescription(typeof(Properties.Resources), "MinLPCOrderDescription")]
        public int MinLPCOrder { get; set; }

        [DefaultValue(-1)]
        [DefaultValueForMode(8, 8, 8, 12, 12, 12, 12, 12, 12, 32, 32, 32)]
        [Browsable(false)]
        [DisplayName("MaxLPCOrder")]
        [SRDescription(typeof(Properties.Resources), "MaxLPCOrderDescription")]
        public int MaxLPCOrder { get; set; }

        [DefaultValue(0)]
        [DisplayName("MinPartitionOrder")]
        [Browsable(false)]
        [SRDescription(typeof(Properties.Resources), "MinPartitionOrderDescription")]
        public int MinPartitionOrder { get; set; }

        [DefaultValue(-1)]
        [DefaultValueForMode(6, 6, 6, 6, 6, 6, 6, 6, 7, 6, 6, 8)]
        [DisplayName("MaxPartitionOrder")]
        [Browsable(false)]
        [SRDescription(typeof(Properties.Resources), "MaxPartitionOrderDescription")]
        public int MaxPartitionOrder { get; set; }

        [DefaultValue(false)]
        [DisplayName("Verify")]
        [SRDescription(typeof(Properties.Resources), "DoVerifyDescription")]
        [JsonProperty]
        public bool DoVerify { get; set; }

        [DefaultValue(true)]
        [DisplayName("MD5")]
        [SRDescription(typeof(Properties.Resources), "DoMD5Description")]
        [JsonProperty]
        public bool DoMD5 { get; set; }

        [DefaultValue(false)]
        [DisplayName("Allow Non-subset")]
        [SRDescription(typeof(Properties.Resources), "AllowNonSubsetDescription")]
        [JsonProperty]
        public bool AllowNonSubset { get; set; }

        [DefaultValue(StereoMethod.Invalid)]
        [DefaultValueForMode(
            /*  0 */ StereoMethod.Independent,
            /*  1 */ StereoMethod.EstimateFixed,
            /*  2 */ StereoMethod.Estimate,
            /*  3 */ StereoMethod.Estimate,
            /*  4 */ StereoMethod.Evaluate,
            /*  5 */ StereoMethod.Evaluate,
            /*  6 */ StereoMethod.Evaluate,
            /*  7 */ StereoMethod.Evaluate,
            /*  8 */ StereoMethod.Evaluate,
            /*  9 */ StereoMethod.Evaluate,
            /* 10 */ StereoMethod.Evaluate,
            /* 11 */ StereoMethod.Evaluate)]
        [Browsable(false)]
        public StereoMethod StereoMethod { get; set; }

        [DefaultValue(PredictionType.None)]
        [DefaultValueForMode(
            /*  0 */ PredictionType.Fixed,
            /*  1 */ PredictionType.Fixed,
            /*  2 */ PredictionType.Levinson,
            /*  3 */ PredictionType.Levinson,
            /*  4 */ PredictionType.Search,
            /*  5 */ PredictionType.Search,
            /*  6 */ PredictionType.Search,
            /*  7 */ PredictionType.Search,
            /*  8 */ PredictionType.Search,
            /*  9 */ PredictionType.Levinson,
            /* 10 */ PredictionType.Search,
            /* 11 */ PredictionType.Search)]
        [Browsable(false)]
        public PredictionType PredictionType { get; set; }

        [DefaultValue(WindowMethod.Invalid)]
        [DefaultValueForMode(
            /*  0 */ WindowMethod.Invalid,
            /*  1 */ WindowMethod.Invalid,
            /*  2 */ WindowMethod.Estimate,
            /*  3 */ WindowMethod.Estimate,
            /*  4 */ WindowMethod.Estimate,
            /*  5 */ WindowMethod.EvaluateN,
            /*  6 */ WindowMethod.EvaluateN,
            /*  7 */ WindowMethod.EvaluateN,
            /*  8 */ WindowMethod.EvaluateN,
            /*  9 */ WindowMethod.EvaluateN,
            /* 10 */ WindowMethod.EvaluateN,
            /* 11 */ WindowMethod.EvaluateN)]
        [Browsable(false)]
        public WindowMethod WindowMethod { get; set; }

        [DefaultValue(WindowFunction.None)]
        [DefaultValueForMode(
            /*  0 */ WindowFunction.None,
            /*  1 */ WindowFunction.None,
            /*  2 */ WindowFunction.Tukey3,
            /*  3 */ WindowFunction.Tukey4,
            /*  4 */ WindowFunction.Tukey4,
            /*  5 */ WindowFunction.Tukey4 | WindowFunction.Tukey3,
            /*  6 */ WindowFunction.Tukey4 | WindowFunction.Tukey3 | WindowFunction.Tukey,
            /*  7 */ WindowFunction.Tukey4 | WindowFunction.Tukey3 | WindowFunction.Tukey2 | WindowFunction.Tukey,
            /*  8 */ WindowFunction.Tukey4 | WindowFunction.Tukey3 | WindowFunction.Tukey2 | WindowFunction.Tukey,
            /*  9 */ WindowFunction.Tukey3 | WindowFunction.Tukey2 | WindowFunction.Tukey,
            /* 10 */ WindowFunction.Tukey3 | WindowFunction.Tukey2 | WindowFunction.Tukey,
            /* 11 */ WindowFunction.Tukey3 | WindowFunction.Tukey2 | WindowFunction.Tukey)]
        [Browsable(false)]
        [DisplayName("WindowFunctions")]
        [SRDescription(typeof(Properties.Resources), "WindowFunctionsDescription")]
        public WindowFunction WindowFunctions { get; set; }

        [DefaultValue(0)]
        [DefaultValueForMode(0, 0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 5)]
        [Browsable(false)]
        public int EstimationDepth { get; set; }

        [DefaultValue(-1)]
        [DefaultValueForMode(1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1)]
        [Browsable(false)]
        public int MinPrecisionSearch { get; set; }

        [DefaultValue(-1)]
        [DefaultValueForMode(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
        [Browsable(false)]
        public int MaxPrecisionSearch { get; set; }

        [DefaultValue(0)]
        [Browsable(false)]
        public int TukeyParts { get; set; }

        [DefaultValue(1.0)]
        [Browsable(false)]
        public double TukeyOverlap { get; set; }

        [DefaultValue(1.0)]
        [Browsable(false)]
        public double TukeyP { get; set; }

        [Browsable(false)]
        public string[] Tags { get; set; }
    }
}
