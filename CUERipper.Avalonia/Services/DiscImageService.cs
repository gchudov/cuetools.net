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
using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Exceptions;
using CUERipper.Avalonia.Extensions;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Processor;
using Microsoft.Extensions.Localization;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public class DiscImageService : IDiscImageService
    {
        public event EventHandler<CUEToolsProgressEventArgs>? OnProgress;
        public event EventHandler<CUEToolsSelectionEventArgs>? OnRepairSelection;

        private readonly ICUEConfigFacade _config;
        private readonly ICUEMetadataStore _metadataStore;
        private readonly IStringLocalizer _localizer;
        private readonly ILogger _logger;

        public DiscImageService(ICUEConfigFacade config
            , ICUEMetadataStore metadataStore
            , IStringLocalizer<Language> stringLocalizer
            , ILogger<DiscImageService> logger)
        {
            _config = config;
            _metadataStore = metadataStore;
            _localizer = stringLocalizer;
            _logger = logger;
        }

        public Task<CUEResult?> ProcessAsync(string cuePath
            , EncodingConfiguration[] encodingConfiguration
            , bool repairable
            , CancellationToken ct)
            => Task.Factory.StartNew<CUEResult?>(() =>
            {
                try
                {
                    return Process(cuePath, encodingConfiguration, repairable, ct);
                }
                catch (StopException)
                {
                    _logger.LogInformation("Processing has been stopped by user.");
                    return CUEResult.Failure(_localizer["Status:RipFailUser"], string.Empty);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Processing has failed! Unexpected error occurred.");
                    return CUEResult.Failure(_localizer["Status:RipFail"]
                        , $"{_localizer["Error:Unexpected"]} {ex.Message}");
                }
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
            // Do not let this task be cancelled by the cancellation token, let it exit cleanly.

        private CUEResult? Process(string cuePath
            , EncodingConfiguration[] encodingConfiguration
            , bool repairable
            , CancellationToken ct)
        {
            if (encodingConfiguration.None())
            {
                throw new ArgumentException("No encoding configuration to continue with."
                    , nameof(encodingConfiguration));
            }

            _config.ApplyEncodingConfiguration(encodingConfiguration[0]);

            CUEResult? result = null;

            if (!_config.SkipRepair && repairable)
            {
                _logger.LogInformation("Start repairing tracks.");

                var repairedCue = Repair(cuePath, encodingConfiguration[0], ct);
                if (repairedCue != null)
                {
                    try
                    {
                        result = new CUEResult(RipStatus.CompletedWithErrors
                            , _localizer["Warning:RipTroubledDisc"]
                            , repairedCue.GenerateVerifyStatus() + "."
                            , cuePath);
                    }
                    finally
                    {
                        repairedCue.Close();
                    }
                }
            } 

            EncodePerConfiguration(cuePath, encodingConfiguration, ct);

            return result;
        }

        private CUESheet? Repair(string cuePath, EncodingConfiguration encodingConfig, CancellationToken ctx)
        {
            var cueSheet = new CUESheet(_config.ToCUEConfig())
            {
                Action = CUEAction.Encode,
                OutputStyle = GetCUEStyle(encodingConfig),
            };

            cueSheet.CUEToolsProgress += (object? sender, CUEToolsProgressEventArgs args) =>
            {
                if (ctx.IsCancellationRequested) throw new StopException();
                OnProgress?.Invoke(sender, args);
            };

            cueSheet.CUEToolsSelection += (object? sender, CUEToolsSelectionEventArgs args) =>
            {
                OnRepairSelection?.Invoke(sender, args);
            };

            cueSheet.Open(cuePath);
            ApplyStoredMetadata(cueSheet);

            cueSheet.UseAccurateRip();

            string cueDirectory = Path.GetDirectoryName(cuePath) ?? throw new DirectoryNotFoundException(cuePath);
            string cueFileName = Path.GetFileName(cuePath);
            string repairPath = $"{cueDirectory}/{Constants.TempFolderCUERipper}";
            string repairCuePath = $"{repairPath}/{cueFileName}";

            if (Directory.Exists(repairPath))
            {
                Directory.Delete(repairPath, true);
            }

            cueSheet.GenerateFilenames(GetEncoderType(encodingConfig), encodingConfig.Encoding, repairCuePath);

            const string REPAIR_SCRIPT = "repair";
            if (!_config.Scripts.TryGetValue(REPAIR_SCRIPT, out CUEToolsScript? value))
            {
                _logger.LogError("Where did the repair script go?");
                throw new CUEToolsCoreException("For some reason the repair script seems to be missing?");
            }

            try
            {
                cueSheet.ExecuteScript(value);

                if (!cueSheet.IsUsingCUEToolsDBFix)
                {
                    // Repair cancelled
                    cueSheet.Close();
                    return null;
                }

                foreach (string source in Directory.GetFiles(repairPath))
                {
                    string fileName = Path.GetFileName(source);
                    string destination = Path.Combine(cueDirectory, fileName);

                    if (File.Exists(destination))
                    {
                        File.Delete(destination);
                    }

                    File.Move(source, destination);
                }
            }
            finally
            {
                if (Directory.Exists(repairPath))
                {
                    Directory.Delete(repairPath, true);
                }
            }

            return cueSheet;
        }

        private void EncodePerConfiguration(string cuePath, EncodingConfiguration[] encodingConfiguration, CancellationToken ct)
        {
            string cueDirectory = Path.GetDirectoryName(cuePath) ?? throw new DirectoryNotFoundException(cuePath);
            string cueFileName = Path.GetFileName(cuePath);

            try
            {
                // Skip the first, because it's already encoded :)
                for (int i = 1; i < encodingConfiguration.Length; ++i)
                {
                    if (ct.IsCancellationRequested) break;

                    var encodingConfig = encodingConfiguration[i];
                    _config.ApplyEncodingConfiguration(encodingConfig);

                    string destination = $"{cueDirectory}/{i}-{encodingConfig.Encoding}";
                    string destinationCuePath = $"{destination}/{cueFileName}";

                    _logger.LogInformation("Start {Encoding} encoding, preset #{Number}.", encodingConfig.Encoding, i);

                    Encode(cuePath, destination, destinationCuePath, encodingConfig, i, encodingConfiguration.Length - 1, ct);

                    _logger.LogInformation("Finished {Encoding} encoding, preset #{Number}.", encodingConfig.Encoding, i);
                }
            }
            finally 
            {
                _config.ApplyEncodingConfiguration(encodingConfiguration[0]);
            }
        }

        private void Encode(string source
            , string destination
            , string destinationCue
            , EncodingConfiguration encodingConfig
            , int current
            , int total
            , CancellationToken ct)
        {
            var cueSheet = new CUESheet(_config.ToCUEConfig())
            {
                Action = CUEAction.Encode,
                OutputStyle = GetCUEStyle(encodingConfig)
            };

            cueSheet.CUEToolsProgress += (object? sender, CUEToolsProgressEventArgs args) =>
            {
                if (ct.IsCancellationRequested) throw new StopException();

                args.status = $"({current}/{total}) {args.status}";

                OnProgress?.Invoke(sender, args);
            };

            cueSheet.Open(source);
            ApplyStoredMetadata(cueSheet);

            var encoderType = GetEncoderType(encodingConfig);

            if (encoderType == AudioEncoderType.Lossless) cueSheet.UseAccurateRip();

            if (Directory.Exists(destination))
            {
                Directory.Delete(destination, true);
            }

            cueSheet.GenerateFilenames(encoderType, encodingConfig.Encoding, destinationCue);

            bool isSuccess = false;

            try
            {
                cueSheet.Go();

                isSuccess = true;
            }
            finally
            {
                if (!isSuccess && Directory.Exists(destination))
                {
                    Directory.Delete(destination, true);
                }

                cueSheet.Close();
            }
        }

        private CUEStyle GetCUEStyle(EncodingConfiguration encodingConfig)
            => encodingConfig.CUEStyleIndex == 0
                ? _config.CanEmbedCUE(encodingConfig) ? CUEStyle.SingleFileWithCUE : CUEStyle.SingleFile
                : CUEStyle.GapsAppended;

        private static AudioEncoderType GetEncoderType(EncodingConfiguration encodingConfig)
            => encodingConfig.IsLossless
                ? AudioEncoderType.Lossless
                : AudioEncoderType.Lossy;

        private void ApplyStoredMetadata(CUESheet cueSheet)
        {
            var stored = _metadataStore.Load(cueSheet.TOC.TOCID);
            if (stored == null)
            {
                _logger.LogWarning("No stored metadata for TOC {TOCID}, continuing with what the CUE provides."
                    , cueSheet.TOC.TOCID);
            }
            else
            {
                cueSheet.CopyMetadata(stored);
            }
        }
    }
}
