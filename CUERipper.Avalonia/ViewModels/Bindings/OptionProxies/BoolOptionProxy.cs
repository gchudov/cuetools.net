#region Copyright (C) 2025 Max Visser
/*
    Copyright (C) 2025 Max Visser

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
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions;

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies
{
    public class BoolOptionProxy : OptionProxy<bool>
    {
        private const string True = "True";
        private const string False = "False";
        public BoolOptionProxy(string name, bool defaultValue, Accessor<bool> accessor)
            : base(name, defaultValue, accessor)
        {
            Options = [True, False];
        }

        protected override string GetStringFromValue(bool val)
            => val ? True : False;

        protected override bool GetValueFromString(string str)
            => str == True;
    }
}
