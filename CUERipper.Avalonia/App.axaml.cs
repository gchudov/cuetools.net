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
using CUERipper.Avalonia.ViewModels;
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
                    .Enrich.WithProperty("Application", "CUERipper")
                    .WriteTo.File("logs/log-.txt"
                        , rollingInterval: RollingInterval.Day
                        , retainedFileCountLimit: 10
                    ).CreateLogger();
            }
        }

        public override void OnFrameworkInitializationCompleted()
        {
            var services = new ServiceCollection();

            // Register services and viewmodels
            ConfigureServices(services);

            var serviceProvider = services.BuildServiceProvider();

            if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            {
                // Avoid duplicate validations from both Avalonia and the CommunityToolkit. 
                // More info: https://docs.avaloniaui.net/docs/guides/development-guides/data-validation#manage-validationplugins
                DisableAvaloniaDataAnnotationValidation();

                desktop.Exit += (sender, args) => { OnApplicationShutdown(serviceProvider); };

                var mainWindow = serviceProvider.GetRequiredService<MainWindow>();
                mainWindow.DataContext = serviceProvider.GetRequiredService<MainWindowViewModel>();

                desktop.MainWindow = mainWindow;
            }

            base.OnFrameworkInitializationCompleted();
        }

        public void ConfigureServices(IServiceCollection services)
        {
            services.AddSingleton<ICUERipperService, CUERipperService>();
            services.AddSingleton<ICUEMetaService, CUEMetaService>();

            if (OS.IsWindows())
                services.AddSingleton<IDriveNotificationService, WindowsDriveNotificationService>();
            else if(OS.IsLinux())
                services.AddSingleton<IDriveNotificationService, LinuxDriveNotificationService>();
            else
                services.AddSingleton<IDriveNotificationService, NullDriveNotificationService>();

            services.AddTransient<MainWindow>();
            services.AddTransient<MainWindowViewModel>();
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
            }

            serviceProvider.Dispose();

            Log.CloseAndFlush();

            if (!Design.IsDesignMode)
            {
                var fileInDir = Directory.GetFiles($"{Constants.PathImageCache}", $"*{Constants.JpgExtension}", SearchOption.TopDirectoryOnly);
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
    }
}