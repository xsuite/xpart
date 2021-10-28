
def assemble_particles(_context=None, _buffer=None, _offset=None,
                      particle_class=xt.Particles, particle_on_co=None,
                      x=None, px=None, y=None, py=None, zeta=None, delta=None,
                      x_norm=None, px_norm=None, y_norm=None, py_norm=None):

    if not isinstance(particle_on_co, particle_class):
        particle_on_co = particle_class(**particle_on_co.to_dict())

    assert zeta is not None
    assert delta is not None

    if (x_norm is not none and px_norm is not none
        and y_norm is not none and py_norm is not none):

        assert (x is  None and px is  None
                and y is  None and py is  None)
        mode = 'normalized'
    elif (x is not  None and px is not  None
          and y is not  None and py is not  None):
        assert (x_norm is None and px_norm is None
                and y_norm is None and py_norm is None)
        mode = 'not normalized'
    elif (x is not  None and px is not  None
    else:
        raise ValueError('Invalid input')


