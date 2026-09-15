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
using CUERipper.Avalonia.Models;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System;
using System.Collections.ObjectModel;
using System.Linq;

namespace CUERipper.Avalonia.ViewModels.UserControls
{
    public partial class EncodingTabContainerViewModel : ViewModelBase
    {
        public ObservableCollection<EncodingSectionViewModel> Tabs { get; } = [];

        [ObservableProperty]
        private bool isReadOnly;

        [ObservableProperty]
        private EncodingSectionViewModel? selectedTab;

        private bool CanAddTab => Tabs.Count < Constants.MaxEncodingTabs;
        private bool CanRemoveTab => Tabs.Count > Constants.MinEncodingTabs;

        private readonly ICUEConfigFacade _config;
        private readonly ILogger _logger;
        private readonly Func<EncodingSectionViewModel> _encodingSectionFactory;

        public EncodingTabContainerViewModel(ICUEConfigFacade config
            , Func<EncodingSectionViewModel> encodingSectionFactory
            , ILogger<EncodingTabContainerViewModel> logger)
        {
            _config = config;
            _logger = logger;
            _encodingSectionFactory = encodingSectionFactory;
        }

        internal void InitializeTabs()
        {
            if (Tabs.Any()) return;

            EncodingConfiguration[] encodingConfig = [];
            try
            {
                if (!string.IsNullOrWhiteSpace(_config.EncodingConfiguration))
                {
                    encodingConfig = JsonConvert.DeserializeObject<EncodingConfiguration[]>(_config.EncodingConfiguration)
                        ?? encodingConfig;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to parse encoding configuration: {Config}", _config.EncodingConfiguration);
            }

            if (encodingConfig.Length == 0)
            {
                CreateTab(null);
            }
            else
            {
                for (int i = 0; i < encodingConfig.Length; ++i)
                {
                    // Skip the first one, read it from the shared settings (CUERipper old)
                    CreateTab(i == 0 ? null : encodingConfig[i]);
                }
            }

            SelectedTab = Tabs.FirstOrDefault();
        }

        internal void PersistTabs()
            => _config.EncodingConfiguration = JsonConvert.SerializeObject(GetEncodingConfigurations());

        public EncodingConfiguration[] GetEncodingConfigurations()
            => Tabs.Select(t => t.GetConfiguration())
                .OfType<EncodingConfiguration>()
                .ToArray();

        [RelayCommand(CanExecute = nameof(CanAddTab))]
        private void AddTab()
            => SelectedTab = CreateTab(null);

        [RelayCommand(CanExecute = nameof(CanRemoveTab))]
        private void RemoveTab()
        {
            if (SelectedTab == null) return;

            int index = Tabs.IndexOf(SelectedTab);
            Tabs.RemoveAt(index);

            SelectedTab = Tabs[Math.Min(index, Tabs.Count - 1)];

            OnTabsChanged();
        }

        private EncodingSectionViewModel CreateTab(EncodingConfiguration? encodingConfig)
        {
            var tab = _encodingSectionFactory();
            if (encodingConfig != null)
            {
                tab.SetConfiguration(encodingConfig);
            }

            Tabs.Add(tab);
            OnTabsChanged();

            return tab;
        }

        private void OnTabsChanged()
        {
            for (int i = 0; i < Tabs.Count; ++i)
            {
                Tabs[i].SectionHeader = i == 0 ? Constants.FirstEncodingTabHeader : i.ToString();
            }

            AddTabCommand.NotifyCanExecuteChanged();
            RemoveTabCommand.NotifyCanExecuteChanged();
        }
    }
}
