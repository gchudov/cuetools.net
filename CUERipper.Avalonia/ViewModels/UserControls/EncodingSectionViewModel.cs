#region Copyright (C) 2026 Max Visser
/*
    Copyright (C) 2026 Max Visser

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <https://www.gnu.org/licenses/>.
*/
#endregion
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Abstractions;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Processor;
using Microsoft.Extensions.Localization;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class EncodingSectionViewModel : ViewModelBase
    {

        public ObservableCollection<string> Compression
        {
            get => [_localizer["Encoding:Lossless"], _localizer["Encoding:Lossy"]];
        }

        [ObservableProperty]
        private string selectedCompression = string.Empty;

        partial void OnSelectedCompressionChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;
            if (string.Compare(oldValue, newValue) == 0) return;

            RefreshEncoding();
        }

        public ObservableCollection<string> CUEStyle
        {
            get => [_localizer["Encoding:Image"], _localizer["Encoding:Tracks"]];
        }

        [ObservableProperty]
        private string selectedCUEStyle = string.Empty;

        public ObservableCollection<string> Encoding { get; set; } = [];

        [ObservableProperty]
        private string selectedEncoding = string.Empty;

        partial void OnSelectedEncodingChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;

            RefreshEncoder();
        }

        public ObservableCollection<string> Encoder { get; set; } = [];

        [ObservableProperty]
        private string selectedEncoder = string.Empty;

        partial void OnSelectedEncoderChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;
            RefreshEncoderSettings();
        }

        [ObservableProperty]
        private int selectedEncoderMode;
        partial void OnSelectedEncoderModeChanged(int oldValue, int newValue)
            => RefreshEncoderModeSelection();

        [ObservableProperty]
        private int encoderModeMaximum;

        [ObservableProperty]
        private bool isEncoderModeEnabled;

        private string[] _encoderModeValues = [];

        [ObservableProperty]
        private string selectedEncoderModeText = string.Empty;

        public string ToolTipCompression { get => _localizer["Encoding:ToolTipCompression"]; }
        public string ToolTipCUEStyle { get => _localizer["Encoding:ToolTipCUEStyle"]; }
        public string ToolTipSettings { get => _localizer["Encoding:ToolTipSettings"]; }
        public string ToolTipEncoderSettings { get => _localizer["Encoding:ToolTipEncoderSettings"]; }

        public IBitmap? IconSettings { get; }
        public IBitmap? IconEncoderSettings { get; }

        [ObservableProperty]
        private string sectionHeader = string.Empty;

        private readonly ICUEConfigFacade _config;
        private readonly IDialogService _dialogService;
        private readonly IStringLocalizer _localizer;
        public EncodingSectionViewModel(ICUEConfigFacade config
            , IDialogService dialogService
            , IIconService iconService
            , IStringLocalizer<Language> localizer)
        {
            _config = config;
            _dialogService = dialogService;
            _localizer = localizer;

            IconSettings = iconService.GetIcon(AppIcon.Cog);
            IconEncoderSettings = iconService.GetIcon(AppIcon.Bolt);

            SelectedCompression = _config.OutputCompression == AudioEncoderType.Lossless
                ? _localizer["Encoding:Lossless"]
                : _localizer["Encoding:Lossy"];

            SelectedCUEStyle = _config.CUEStyleIndex >= 0 && _config.CUEStyleIndex < CUEStyle.Count
                ? CUEStyle[_config.CUEStyleIndex]
                : CUEStyle[CUEStyle.Count - 1];
        }

        public bool IsLossless()
            => SelectedCompression == _localizer["Encoding:Lossless"];

        public void RefreshEncoding()
        {
            Encoding.Clear();
            new ObservableCollection<string>(
                _config.Formats
                    .Where(f => f.Value.allowLossless == IsLossless() || f.Value.allowLossy == !IsLossless())
                    .Select(f => f.Key)
            ).MoveAll(Encoding);

            var favorite = IsLossless() ? _config.DefaultLosslessFormat : _config.DefaultLossyFormat;
            if (!Encoding.Contains(favorite)) favorite = Encoding[0];

            SelectedEncoding = Encoding.Any() ? favorite : string.Empty;
        }

        public void RefreshEncoder()
        {
            Encoder.Clear();

            new ObservableCollection<string>(
                _config.Encoders
                    .Where(e => string.Compare(e.Extension, SelectedEncoding, true) == 0)
                    .Select(e => e.Name)
            ).MoveAll(Encoder);

            var favoriteEncoder = _config.Formats
                .Where(f => f.Key == SelectedEncoding)
                .Select(f => (IsLossless() ? f.Value.encoderLossless?.Name : f.Value.encoderLossy?.Name) ?? string.Empty)
                .FirstOrDefault();

            SelectedEncoder = Encoder.Any()
                ? Encoder.Where(e => e == favoriteEncoder).FirstOrDefault() ?? Encoder[0]
                : string.Empty;
        }

        public void RefreshEncoderSettings()
        {
            var currentEncoding = _config.Formats
                .Where(f => f.Key == SelectedEncoding)
                .Select(e => e.Value)
                .Single();

            if (string.IsNullOrWhiteSpace(SelectedEncoder))
            {
                _encoderModeValues = ["No encoder found for this format."];
                SelectedEncoderMode = 0;
                EncoderModeMaximum = 0;
                return;
            }

            var requestedEncoder = _config.Encoders
                .Where(e => string.Compare(e.Extension, SelectedEncoding, true) == 0)
                .Where(e => string.Compare(e.Name, SelectedEncoder, true) == 0)
                .Single();

            _encoderModeValues = requestedEncoder.SupportedModes.Split(' ');
            EncoderModeMaximum = _encoderModeValues.Length - 1;
            SelectedEncoderMode = EncoderModeMaximum > requestedEncoder.EncoderModeIndex
                ? Math.Max(0, requestedEncoder.EncoderModeIndex)
                : 0;

            IsEncoderModeEnabled = EncoderModeMaximum > 0;
        }

        public void RefreshEncoderModeSelection()
        {
            var encoderMode = _encoderModeValues[SelectedEncoderMode];
            SelectedEncoderModeText = encoderMode;
        }

        [RelayCommand]
        private async Task OpenSettingsAsync()
            => await _dialogService.ShowOptionsAsync();

        [RelayCommand]
        private async Task OpenEncoderSettingsAsync()
        {
            var encoderSettings = _config.Encoders
                .Where(e => string.Compare(e.Name, SelectedEncoder, true) == 0
                    && string.Compare(e.Extension, SelectedEncoding, true) == 0)
                .Select(e => e.Settings)
                .FirstOrDefault();

            if (encoderSettings != null)
            {
                await _dialogService.ShowEncoderOptionsAsync(encoderSettings);
            }
        }

        public EncodingConfiguration? GetConfiguration()
        {
            if (string.IsNullOrWhiteSpace(SelectedEncoder)) return null;

            var requestedEncoder = _config.Encoders
                .Where(e => string.Compare(e.Extension, SelectedEncoding, true) == 0)
                .Where(e => string.Compare(e.Name, SelectedEncoder, true) == 0)
                .SingleOrDefault();

            if (requestedEncoder == null) return null;

            return new EncodingConfiguration
            (
                IsLossless: IsLossless()
                , CUEStyleIndex: CUEStyle.IndexOf(SelectedCUEStyle)
                , Encoding: SelectedEncoding
                , Encoder: SelectedEncoder
                , EncoderMode: _encoderModeValues[SelectedEncoderMode]
            );
        }

        public void SetConfiguration(EncodingConfiguration encodingConfiguration)
        {
            SelectedCompression = encodingConfiguration.IsLossless
                ? _localizer["Encoding:Lossless"]
                : _localizer["Encoding:Lossy"];

            SelectedCUEStyle = encodingConfiguration.CUEStyleIndex >= 0 && encodingConfiguration.CUEStyleIndex < CUEStyle.Count
                ? CUEStyle[encodingConfiguration.CUEStyleIndex]
                : CUEStyle[CUEStyle.Count - 1];

            SelectedEncoding = encodingConfiguration.Encoding;
            SelectedEncoder = encodingConfiguration.Encoder;

            int encoderModeIndex = Array.IndexOf(_encoderModeValues, encodingConfiguration.EncoderMode);
            SelectedEncoderMode = Math.Max(0, encoderModeIndex);
        }
    }
}
