// SPDX-License-Identifier: GPL-2.0-or-later
using System;

internal static class Program
{
    private static int Main()
    {
        try
        {
            int count = 0;
            OffsetCalibrationTests.Run((condition, message) =>
            {
                if (!condition) throw new Exception(message);
                count++;
            });
            Console.WriteLine("Passed " + count + " Windows offset calibration checks.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
