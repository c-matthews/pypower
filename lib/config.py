import ConfigParser
import os

class IniError( ConfigParser.Error ):
    pass

class IniFile( ConfigParser.ConfigParser ):

    def __init__(self, filename):
        
        ConfigParser.ConfigParser.__init__(self)
        
        if filename is not None:
            if not os.path.exists(filename):
                raise IOError("Unable to open configuration file %s." % (filename, ))
            self.read(filename)
        else:
            raise IOError("No filename specified.")

    def get(self, section, name, default=None):
        try:
            return ConfigParser.ConfigParser.get(self, section, name)
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError) as e:
            if default is None:
                raise IniError("Cannot find an option '%s' in the '[%s]' section in the ini file. "%(name,section))
            else:
                return default

    # these functions override the default parsers to allow for extra formats
    def getint(self, section, name, default=None):
        try:
            return ConfigParser.ConfigParser.getint(self, section, name)
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, IniError) as e:
            if default is None:
                raise IniError("Cannot find an integer '%s' in the '[%s]' section in the ini file. "%(name,section))
            elif not isinstance(default, int):
                raise TypeError("Default not integer")
            else:
                return default

    def getfloat(self, section, name, default=None):
        try:
            return ConfigParser.ConfigParser.getfloat(self, section, name)
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, IniError) as e:
            if default is None:
                raise IniError("Cannot find a float '%s' in the '[%s]' section in the ini file. "%(name,section))
            elif not isinstance(default, float):
                raise TypeError("Default not float")
            else:
                return default

    def getboolean(self, section, name, default=None):
        try:
            return ConfigParser.ConfigParser.getboolean(self, section, name)
        except ValueError:
            # additional options t/y/n/f
            value = self.get(section, name).lower()
            if value in ['y', 'yes', 't','true']:
                return True
            elif value in ['n', 'no', 'f','false']:
                return False
            else:
                raise ValueError("Unable to parse parameter %s = %s into boolean form"% (name, value))
        except (ConfigParser.NoSectionError, ConfigParser.NoOptionError, IniError) as e:
            if default is None:
                raise IniError("Cannot find a boolean '%s' in the '[%s]' section in the ini file. "%(name,section))
            elif not isinstance(default, bool):
                raise TypeError("Default not boolean")
            else:
                return default