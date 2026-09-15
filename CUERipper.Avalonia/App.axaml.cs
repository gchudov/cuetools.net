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
using Avalonia.Data.Core.Plugins;
using Avalonia.Markup.Xaml;
using CUERipper.Avalonia.Compatibility;
using CUERipper.Avalonia.Configuration;
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Services;
using CUERipper.Avalonia.Services.Abstractions;
using CUERipper.Avalonia.Utilities;
using CUERipper.Avalonia.ViewModels;
using CUERipper.Avalonia.ViewModels.UserControls;
using CUERipper.Avalonia.Views;
using Microsoft.Extensions.DependencyInjection;
using Serilog;
using Serilog.Events;
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;

namespace CUERipper.Avalonia
{
    public partial class App : Application
    {
        public override void Initialize()
        {
            AvaloniaXamlLoader.Load(this);

            if (Design.IsDesignMode)
            {
                Log.Logger = new LoggerConfiguration().CreateLogger();
            }
            else
            {
                Log.Logger = new LoggerConfiguration()
                    .MinimumLevel.Debug()
                    .MinimumLevel.Override("Microsoft", LogEventLevel.Warning)
                    .Enrich.FromLogContext()
                    .Enrich.WithProperty("Application", Constants.ApplicationName)
                    .WriteTo.File(Path.Combine(Constants.ProfileDir, "logs/log-.txt")
                        , rollingInterval: RollingInterval.Day
                        , retainedFileCountLimit: 10
                    ).CreateLogger();
            }

            LibraryResolver.Init();
        }

        public override void OnFrameworkInitializationCompleted()
        {
            var services = new ServiceCollection();

            EnsureUserDirectoryExists();

            // Register services and viewmodels
            ConfigureServices(services);

            var serviceProvider = services.BuildServiceProvider();

            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                // Avoid duplicate validations from both Avalonia and the CommunityToolkit. 
                // More info: https://docs.avaloniaui.net/docs/guides/development-guides/data-validation#manage-validationplugins
                DisableAvaloniaDataAnnotationValidation();

                desktop.Exit += (sender, args) => { OnApplicationShutdown(serviceProvider); };

                desktop.MainWindow = serviceProvider.GetRequiredService<MainWindow>();
            }

            base.OnFrameworkInitializationCompleted();
        }

        public void ConfigureServices(IServiceCollection services)
        {
			// Thin wrappers around static CUETools functions
            services.AddSingleton<ICUEMetadataStore, CUEMetadataStore>();
            services.AddSingleton<ICDDriveEnumerator, CDDriveEnumerator>();
            services.AddSingleton<ICDRipperFactory, CDRipperFactory>();
            services.AddSingleton<IRemoteMetadataLookup, RemoteMetadataLookup>();
            
            services.AddSingleton<IDiscRippingService, DiscRippingService>();
            services.AddSingleton<IDiscImageService, DiscImageService>();
            services.AddSingleton<IAlbumMetadataService, AlbumMetadataService>();

            if (OS.IsWindows())
                services.AddSingleton<IDriveNotificationService, WindowsDriveNotificationService>();
            else if(OS.IsLinux())
                services.AddSingleton<IDriveNotificationService, LinuxDriveNotificationService>();
            else
                services.AddSingleton<IDriveNotificationService, NullDriveNotificationService>();

            services.AddTransient<MainWindow>();
            services.AddTransient<MainWindowViewModel>();
            services.AddTransient<DriveSettingSectionViewModel>();
            services.AddTransient<EncodingTabContainerViewModel>();
            services.AddTransient<EncodingSectionViewModel>();
            services.AddTransient<Func<EncodingSectionViewModel>>(sp
                => () => sp.GetRequiredService<EncodingSectionViewModel>());
            services.AddTransient<MetaGridViewModel>();
            services.AddTransient<TrackGridViewModel>();
            services.AddTransient<CoverViewerViewModel>();
            services.AddTransient<RipSessionViewModel>();
            services.AddTransient<MessageBox>();
            services.AddTransient<MessageBoxViewModel>();
            services.AddTransient<PathFormatDialog>();
            services.AddTransient<PathFormatDialogViewModel>();
            services.AddTransient<UpdateDialog>();
            services.AddTransient<UpdateDialogViewModel>();
            services.AddTransient<OptionsDialog>();
            services.AddTransient<OptionsDialogViewModel>();
            services.AddTransient<EncoderOptionsDialog>();
            services.AddTransient<EncoderOptionsDialogViewModel>();
            services.AddTransient<RepairSelectionDialog>();
            services.AddTransient<RepairSelectionDialogViewModel>();

            services.AddSingleton<IUIDispatcher, UIDispatcher>();

            services.AddTransient<IDialogService, DialogService>();
            services.AddSingleton<IBitmapFactory, AvaloniaBitmapFactory>();
            services.AddSingleton<IIconService, IconService>();

            services.AddLogging(builder =>
            {
                builder.AddSerilog();
            });

            var config = CUEConfigFacade.Create();
            services.AddSingleton<ICUEConfigFacade>(config);

            services.AddSingleton<HttpClient>(CreateHttpClient(config));
            services.AddSingleton<IUpdateService, UpdateService>();

            services.AddLocalization(options => options.ResourcesPath = "Resources");
            Thread.CurrentThread.CurrentUICulture = CultureInfo.GetCultureInfo(config.Language);
        }

        private static HttpClient CreateHttpClient(CUEConfigFacade config)
        {
            HttpClient? httpClient = null;

            var proxy = config.ToCUEConfig().GetProxy();
            if (proxy != null)
            {
                Uri cueToolsUri = new("https://cue.tools/");
                Uri? proxyUri = proxy.GetProxy(cueToolsUri);
                if (proxyUri != null && proxyUri != cueToolsUri)
                {
                    var handler = new HttpClientHandler
                    {
                        Proxy = proxy
                        , UseProxy = true
                    };

                    httpClient = new HttpClient(handler);
                }
            }

            httpClient ??= new HttpClient();
            httpClient.DefaultRequestHeaders.UserAgent.ParseAdd(Constants.UserAgent);
            return httpClient;
        }

        private static void OnApplicationShutdown(ServiceProvider serviceProvider)
        {
            if (!Design.IsDesignMode)
            {
                var config = serviceProvider.GetRequiredService<ICUEConfigFacade>();
                config.Save();

                // Save unpersisted metadata changes
                try
                {
                    serviceProvider.GetRequiredService<IAlbumMetadataService>().Save();
                }
                catch (Exception ex)
                {
                    Log.Error(ex, "Failed to save album metadata during shutdown.");
                }
            }

            serviceProvider.Dispose();

            Log.CloseAndFlush();

            if (!Design.IsDesignMode && Directory.Exists(Constants.PathImageCache))
            {
                var fileInDir = Directory.GetFiles(Constants.PathImageCache, $"*{Constants.JpgExtension}", SearchOption.TopDirectoryOnly);
                foreach (var file in fileInDir)
                {
                    File.Delete(file);
                }
            }
        }

        private void DisableAvaloniaDataAnnotationValidation()
        {
            // Get an array of plugins to remove
            var dataValidationPluginsToRemove =
                BindingPlugins.DataValidators.OfType<DataAnnotationsValidationPlugin>().ToArray();

            // remove each entry found
            foreach (var plugin in dataValidationPluginsToRemove)
            {
                BindingPlugins.DataValidators.Remove(plugin);
            }
        }

        private void EnsureUserDirectoryExists()
        {
            if (!Directory.Exists(Constants.ProfileDir))
            {
                Directory.CreateDirectory(Constants.ProfileDir);
            }
        }
    }
}