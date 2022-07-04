//
//
//                                  DeviceChooser.h                                                                                                                 Asian_Test.h
//
//
// (c) Mark Joshi 2014
// This code is released under the GNU public licence version 3

/*

The purpose of this file is strategize device choice.
*/
#ifndef DEVICE_CHOOSER_H
#define DEVICE_CHOOSER_H

class DeviceChooser
{
public:

    DeviceChooser(){}

    virtual int WhichDevice() const =0;

    virtual ~DeviceChooser(){}

};


class DeviceChooserMax : public DeviceChooser
{
public:

    DeviceChooserMax(){}

    virtual int WhichDevice() const;

  

};


class DeviceChooserSpecific : public DeviceChooser
{
public:

    DeviceChooserSpecific(int dev);

    virtual int WhichDevice() const;

private:
    int dev_;

};
#endif
