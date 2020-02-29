#pragma once
#include "Token.h"

//Transforms unary operators into function form (eg. -x to neg(x)), so that they can be converted to prefix.
void unaryOpToFunction(TokenList& tl);

//Converts juxtaposed multiplication to explicit multiplication (eg. 5z to 5*z)
void juxtaposeToExplicit(TokenList& tl);