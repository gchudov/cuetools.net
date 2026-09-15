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
using CUERipper.Avalonia.Models;
using CUETools.Processor;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services.Abstractions
{
    public interface IDiscImageService
    {
        /// <summary>
        /// Fired from non UI thread
        /// </summary>
        public event EventHandler<CUEToolsProgressEventArgs>? OnProgress;
        /// <summary>
        /// Fired from non UI thread
        /// </summary>
        public event EventHandler<CUEToolsSelectionEventArgs>? OnRepairSelection;

        Task<CUEResult?> ProcessAsync(string cuePath
            , EncodingConfiguration[] encodingConfiguration
            , bool repairable
            , CancellationToken token);
    }
}
