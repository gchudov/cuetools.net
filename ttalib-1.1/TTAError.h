/*
 * TTAError.h
 *
 * Description: Errors processor internal interface
 *
 * Copyright (c) 2004 Alexander Djourik. All rights reserved.
 * Copyright (c) 2004 Pavel Zhilin. All rights reserved.
 *
 */

/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * aint with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * Please see the file COPYING in this directory for full copyright
 * information.
 */

#pragma once
#include <stdexcept>

namespace TTALib 
{
	enum TTAError 
	{
		TTA_NO_ERROR = 0, 
		FORMAT_ERROR,
		FILE_ERROR,
		FIND_ERROR,
		CREATE_ERROR,
		OPEN_ERROR,
		WRITE_ERROR,
		READ_ERROR,
		MEMORY_ERROR,
		TTA_CANCELED
	};

	class TTAException :
		public std::exception
	{
		TTAError errNo;

	public:		
		TTAException(TTAError err) : errNo (err) {}

		TTAException(const TTAException &ex)
		{
			if (&ex != this) errNo = ex.errNo;
		}

		TTAError GetErrNo () { return errNo; }

		virtual ~TTAException(void) {}
	};
};

