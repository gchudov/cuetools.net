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
using Microsoft.Extensions.Localization;
using System.Collections.ObjectModel;
using System.Linq;

namespace CUERipper.Avalonia.ViewModels
{
    public partial class PathFormatDialogViewModel : ViewModelBase
    {
        [ObservableProperty]
        private bool readOnly;

        [ObservableProperty]
        private bool maximumReached;

        public ObservableCollection<string> Formats { get; } = [];

        [ObservableProperty]
        private int formatIndex = -1;
        partial void OnFormatIndexChanged(int oldValue, int newValue)
        {
            if (oldValue == newValue) return;
            if (newValue == -1)
            {
                // Workaround, as editing removes existing string in the observable collection.
                FormatIndex = oldValue < Formats.Count ? oldValue : 0;
                return;
            }

            ReadOnly = newValue < Constants.DefaultPathFormats.Length;
            FormatText = Formats[newValue];
        }

        [ObservableProperty]
        private string outputPreview = string.Empty;

        [ObservableProperty]
        private string formatText = string.Empty;
        partial void OnFormatTextChanged(string? oldValue, string newValue)
        {
            if (string.IsNullOrWhiteSpace(newValue)) return;
            if (string.Compare(oldValue, newValue) == 0) return;

            Formats[FormatIndex] = newValue;

            try
            {
                OutputPreview = _meta.PathStringFromFormat(newValue, _config);
            }
            catch
            {
                OutputPreview = _localizer["PathFormat:ParseError"];
            }
        }

        [ObservableProperty]
        // Default is null, triggers on true or false
        private bool? affirmative;

        private AlbumMetadata? _meta;

        public IBitmap? IconNew { get; }
        public IBitmap? IconCopy { get; }
        public IBitmap? IconDelete { get; }

        private readonly ICUEConfigFacade _config;
        private readonly IStringLocalizer _localizer;
        public PathFormatDialogViewModel(ICUEConfigFacade config
            , IIconService iconService
            , IStringLocalizer<Language> localizer)
        {
            _config = config;
            _localizer = localizer;

            IconNew = iconService.GetIcon(AppIcon.Add);
            IconCopy = iconService.GetIcon(AppIcon.Multiply);
            IconDelete = iconService.GetIcon(AppIcon.Subtract);

            Formats.CollectionChanged += (_, _) => MaximumReached
                = Formats.Count - Constants.DefaultPathFormats.Length >= Constants.MaxPathFormats;
        }

        public void SetMetadata(AlbumMetadata? meta)
        {
            _meta = meta;

            new ObservableCollection<string>(
                Constants.DefaultPathFormats
                    .Concat(_config.PathFormatTemplates)
            ).MoveAll(Formats);

            var index = Formats.IndexOf(_config.PathFormat);
            if (index == -1)
            {
                index = Constants.DefaultPathFormats.Length;
                Formats.Insert(index, _config.PathFormat);
            }

            FormatIndex = index;
        }

        [RelayCommand]
        private void NewFormat()
        {
            Formats.Add(string.Empty);
            FormatIndex = Formats.Count - 1;
        }

        [RelayCommand]
        private void CopyFormat()
        {
            Formats.Add(Formats[FormatIndex]);
            FormatIndex = Formats.Count - 1;
        }

        [RelayCommand]
        private void DeleteFormat()
        {
            FormatIndex -= 1;
            Formats.RemoveAt(FormatIndex + 1);
        }

        [RelayCommand]
        private void Confirm()
        {
            var selectedFormat = Formats[FormatIndex];
            if (ContainsVariable(selectedFormat)) _config.PathFormat = selectedFormat;

            _config.PathFormatTemplates = Formats
                .Skip(Constants.DefaultPathFormats.Length)
                .Where(ContainsVariable)
                .ToList();

            Affirmative = true;
        }

        [RelayCommand]
        private void Cancel() => Affirmative = false;

        private static bool ContainsVariable(string format)
            => format.Contains("%");
    }
}