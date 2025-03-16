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
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions;

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies
{
    public class IntOptionProxy : OptionProxy<int>
    {
        private readonly int? _minValue;
        private readonly int? _maxValue;

        public IntOptionProxy(string name, int defaultValue, Accessor<int> accessor)
            : base(name, defaultValue, accessor)
        {
        }

        public IntOptionProxy(string name, int defaultValue, int minValue, int maxValue, Accessor<int> accessor)
            : base(name, defaultValue, accessor)
        {
            _minValue = minValue;
            _maxValue = maxValue;
        }

        protected override string GetStringFromValue(int val)
            => val.ToString();

        protected override int GetValueFromString(string str)
            => int.TryParse(str, out int result) ? result : Default;

        protected override int ContainWithinRange(int val)
            => _minValue != null && _maxValue != null
                ? MathClamp.Clamp(val, _minValue.Value, _maxValue.Value)
                : val;
    }
}
