#include "pend.h"

int main(int argc, char *argv[])
{
  double st[6] = {0, 0, 60 * (2 * 3.14) / 360, 0, 0, 0};
  doublePole *Pole = new doublePole(st);

  for (int i = 0; i < 100; i++)
  {
    Pole->step(10, st);
    Pole->print();
  }

  return 0;
}