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
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.Views;
using CUETools.Codecs;
using CUETools.Processor;
using Microsoft.Extensions.DependencyInjection;
using System;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public sealed class DialogService : IDialogService
    {
        private readonly IServiceProvider _serviceProvider;

        private static Window Owner
        {
            get
            {
                var app = Application.Current?.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime
                    ?? throw new NotInitializedException(nameof(Owner));

                // Naive method to get the latest window
                // In theory it could be another locked window that can create a new dialog, 
                // but currently that would be unexpected behavior.
                return app.Windows.Count > 1
                    ? app.Windows[app.Windows.Count - 1]
                    : app.MainWindow ?? throw new NotInitializedException(nameof(Owner));
            }
        }

        public DialogService(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public async Task ShowOptionsAsync()
        {
            var optionsDialog = _serviceProvider.GetRequiredService<OptionsDialog>();

            await optionsDialog.CreateDialogAsync(Owner);
        }

        public async Task ShowEncoderOptionsAsync(IAudioEncoderSettings encoderSettings)
        {
            var encoderOptionsDialog = _serviceProvider.GetRequiredService<EncoderOptionsDialog>();

            await encoderOptionsDialog.CreateDialogAsync(Owner, encoderSettings);
        }

        public async Task ShowPathFormatAsync(AlbumMetadata? meta)
        {
            var pathFormatDialog = _serviceProvider.GetRequiredService<PathFormatDialog>();

            // TODO Consider handling the return value
            _ = await pathFormatDialog.CreateDialogAsync(Owner, meta);
        }

        public async Task<bool> ShowUpdateAsync()
        {
            using var scope = _serviceProvider.CreateScope();
            var updateDialog = scope.ServiceProvider.GetRequiredService<UpdateDialog>();

            return await updateDialog.CreateDialogAsync(Owner);
        }

        public async Task<bool> ShowMessageAsync(MessageBoxDefinition definition)
        {
            var messageBox = _serviceProvider.GetRequiredService<MessageBox>();

            return await messageBox.CreateDialogAsync(Owner, definition);
        }

        public async Task<int> ShowRepairSelectionAsync(CUEToolsSourceFile[] sourceFiles)
        {
            var repairSelectionDialog = _serviceProvider.GetRequiredService<RepairSelectionDialog>();

            return await repairSelectionDialog.CreateDialogAsync(Owner, sourceFiles);
        }
    }
}
