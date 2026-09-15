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
namespace CUERipper.Avalonia.Models
{
    public sealed class CUEResult
    {
        public RipStatus Status { get; }
        public string StatusText { get; }
        public string PopupContent { get; }
        public string CUEPath { get; } = string.Empty;

        public bool IsSuccess { get => Status != RipStatus.Failed; }

        private CUEResult(RipStatus status, string statusText, string popupContent)
        {
            Status = status;
            StatusText = statusText;
            PopupContent = popupContent;
        }

        internal CUEResult(RipStatus status
            , string statusText
            , string popupContent
            , string cuePath)
            : this(status, statusText, popupContent)
        {
            CUEPath = cuePath;
        }

        public static CUEResult Failure(string statusText, string popupContent)
            => new(RipStatus.Failed, statusText, popupContent);
    }
}
