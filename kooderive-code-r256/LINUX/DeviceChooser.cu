//
//
//                                  DeviceChooser.cu                                                                                                               Asian_Test.h
//
//
// (c) Mark Joshi 2014
// This code is released under the GNU public licence version 3

/*

The purpose of this file is strategize device choice.
*/

#include "DeviceChooser.h"
#include <cutil.h>

int DeviceChooserMax::WhichDevice() const
{
    return cutGetMaxGflopsDeviceId();  

}


DeviceChooserSpecific::DeviceChooserSpecific(int dev) : dev_(dev)
{
}


int DeviceChooserSpecific::WhichDevice() const
{
   return dev_;
}
