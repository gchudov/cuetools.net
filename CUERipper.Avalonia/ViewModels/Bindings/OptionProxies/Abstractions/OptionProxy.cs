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
using CommunityToolkit.Mvvm.Input;
using CUERipper.Avalonia.Utilities;
using System;
using System.Collections.Generic;
using System.ComponentModel;

namespace CUERipper.Avalonia.ViewModels.Bindings.OptionProxies.Abstractions
{
    public abstract class OptionProxy<T> : IOptionProxy
    {
        public string Name { get; set; }
        public T Default { get; init; }
        private Accessor<T> Accessor { get; init; }

        public bool IsCombo => Options.Count != 0;
        public bool IsReadOnly => Accessor.IsReadOnly;

        protected abstract string GetStringFromValue(T val);
        protected abstract T GetValueFromString(string str);
        protected virtual T ContainWithinRange(T value) => value;

        private string _viewValue = string.Empty;
        public string Value
        {
            get
            {
                try
                {
                    _viewValue = GetStringFromValue(Accessor.Get());
                }
                catch (TypeInitializationException ex)
                    when (ex.InnerException != null && ex.InnerException.GetType() == typeof(DllNotFoundException))
                {
                    // CUETools error
                    _viewValue = "Dll not found!";
                }

                return _viewValue ?? string.Empty;
            }
            set
            {
                if (value == null || IsReadOnly) return;

                try
                {
                    T actualValue = GetValueFromString(value);
                    actualValue = ContainWithinRange(actualValue);

                    _viewValue = GetStringFromValue(actualValue);
                    Accessor.Set(actualValue);
                }
                catch (TypeInitializationException ex)
                    when (ex.InnerException != null && ex.InnerException.GetType() == typeof(DllNotFoundException))
                {
                    // CUETools error
                    // do nothing...
                }
            }
        }

        public List<string> Options { get; set; } = [];
        public IRelayCommand ResetCommand { get; }
        public OptionProxy(string name, T defaultValue, Accessor<T> accessor)
        {
            Name = name;
            Default = defaultValue;
            Accessor = accessor;

            ResetCommand = new RelayCommand(() =>
            {
                accessor.Set(Default);
                OnPropertyChanged(nameof(Value));
            });
        }

        public event PropertyChangedEventHandler? PropertyChanged;
        protected virtual void OnPropertyChanged(string propertyName)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
    }
}
