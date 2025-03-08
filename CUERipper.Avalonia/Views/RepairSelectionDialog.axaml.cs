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
using Avalonia.Controls;
using Avalonia.Interactivity;
using CUERipper.Avalonia.Extensions;
using CUETools.Processor;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace CUERipper.Avalonia;

public partial class RepairSelectionDialog : Window
{
    public required CUEToolsSourceFile[] SourceFiles { get; init; }
    public int Selection { get; set; } = -1;
    public string[] Values { get; set; } = [];

    public RepairSelectionDialog()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
        {
            const string DUMMY = "Time 1\nTime 2\nTime 3\nTime 4\nTime 5\nTime 6\nTime 7\nTime 8\nTime 9\nTime 10\n";
            string[] options = ["Repair option 1", "Repair option 2", "Repair option 3"];
            SourceFiles = new CUEToolsSourceFile[options.Length];
            for(int i = 0; i < options.Length; ++i)
            {
                using TextReader reader = new StringReader(DUMMY);
                SourceFiles[i] = new(options[i], reader);
            }

            Selection = 0;
            Init();
        }
    }

    private void Init()
    {
        Values = SourceFiles.Select(s => s.path).ToArray();
        repairSelection.ItemsSource = Values;
        repairSelection.SelectionChanged += OnSelectionChanged;
        repairSelection.SelectedIndex = 0;
    }

    public void OnSelectionChanged(object? sender, SelectionChangedEventArgs args)
    {
        var index = repairSelection.SelectedIndex;
        description.Text = SourceFiles[index].contents;
    }

    private void OnOkClicked(object? sender, RoutedEventArgs e)
    {
        Selection = repairSelection.SelectedIndex;
        Close();
    }

    private void OnCancelClicked(object? sender, RoutedEventArgs e)
    {
        Selection = -1;
        Close();
    }

    public static async Task<int> CreateAsync(Window owner, CUEToolsSourceFile[] sourceFiles)
    {
        var repairSelectionWindow = new RepairSelectionDialog()
        {
            Owner = owner
            , SourceFiles = sourceFiles
        };

        repairSelectionWindow.Init();
        await repairSelectionWindow.ShowDialog(owner, lockParent: true);

        return repairSelectionWindow.Selection;
    }
}