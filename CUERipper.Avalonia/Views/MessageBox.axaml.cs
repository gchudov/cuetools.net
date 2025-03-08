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
using CUERipper.Avalonia.ViewModels;
using Microsoft.Extensions.Localization;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Views;

public partial class MessageBox : Window
{
    public enum MessageBoxType 
    {
        Ok
        , YesNo
        , OkCancel
    }

    public bool Affirmative { get; set; }

    public MessageBox()
    {
        InitializeComponent();

        if (Design.IsDesignMode)
        {
            Title = "CUERipper Messagebox";
            Design.SetDataContext(this, new MessageBoxViewModel() 
            {
                Message = @"Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Fusce eu magna ut turpis faucibus gravida. Maecenas interdum urna non eros varius, ac rhoncus nibh consectetur. 
Ut ligula mauris, viverra nec maximus quis, convallis at turpis. Curabitur in dictum magna, ut rhoncus orci. 
Nulla facilisi. In sit amet metus tellus. Suspendisse ut leo eget tortor auctor rhoncus a in purus. 
Pellentesque laoreet tempor nisi, nec pharetra odio facilisis vel. Nullam hendrerit nulla sit amet enim volutpat mollis. 
Sed faucibus sem eu turpis blandit tempor. Vestibulum sagittis vehicula lorem, pretium sagittis nisl consectetur vel. 
Phasellus placerat pharetra sem, non faucibus neque sagittis at. Mauris luctus varius lectus at viverra. 
Vestibulum sed odio nibh."
                , Affirm = "Yes"
                , Negate = "Not Yes"
                , ShowNegate = true
            });
        }
    }

    public void SetMessage(string message)
    {
        if (DataContext is MessageBoxViewModel viewModel)
        {
            viewModel.Message = message;
        }
    }

    public void SetType(MessageBoxType type, IStringLocalizer localizer)
    {
        if (DataContext is MessageBoxViewModel viewModel)
        {
            switch (type)
            {
                case MessageBoxType.Ok:
                    {
                        viewModel.Affirm = localizer["Generic:Ok"];
                    } break;
                case MessageBoxType.YesNo:
                    {
                        viewModel.Affirm = localizer["Generic:Yes"];
                        viewModel.Negate = localizer["Generic:No"];
                        viewModel.ShowNegate = true;
                    } break;
                case MessageBoxType.OkCancel:
                    {
                        viewModel.Affirm = localizer["Generic:Ok"];
                        viewModel.Negate = localizer["Generic:Cancel"];
                        viewModel.ShowNegate = true;
                    } break;
            }
        }
    }

    private void OnAffirmClicked(object? sender, RoutedEventArgs e)
    {
        Affirmative = true;

        Close();
    }

    private void OnNegateClicked(object? sender, RoutedEventArgs e)
    {
        Affirmative = false;

        Close();
    }

    public static async Task<bool> CreateDialogAsync(string title
        , string message
        , Window owner
        , IStringLocalizer localizer
        , MessageBoxType type = MessageBoxType.Ok)
    {
        var messageBox = new MessageBox()
        {
            Owner = owner
            , Title = string.IsNullOrWhiteSpace(title) ? "MessageBox" : title
            , DataContext = new MessageBoxViewModel()
        };

        messageBox.SetMessage(message);
        messageBox.SetType(type, localizer);

        await messageBox.ShowDialog(owner, lockParent: true);

        return messageBox.Affirmative;
    }
}