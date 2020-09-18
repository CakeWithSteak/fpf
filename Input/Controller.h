#pragma once

class Controller {
public:
    virtual bool process(double dt) = 0;
};