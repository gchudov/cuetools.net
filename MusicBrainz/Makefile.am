ASSEMBLY = MusicBrainz.dll
REFERENCES =  \
	-r:System \
	-r:System.Xml
SOURCES =  \
	MusicBrainz/Artist.cs \
	MusicBrainz/Disc.cs \
	MusicBrainz/DiscLinux.cs \
	MusicBrainz/DiscWin32.cs \
	MusicBrainz/DiscWin32NT.cs \
	MusicBrainz/Event.cs \
	MusicBrainz/Label.cs \
	MusicBrainz/LocalDisc.cs \
	MusicBrainz/MusicBrainzEntity.cs \
	MusicBrainz/MusicBrainzException.cs \
	MusicBrainz/MusicBrainzItem.cs \
	MusicBrainz/MusicBrainzObject.cs \
	MusicBrainz/MusicBrainzService.cs \
	MusicBrainz/Query.cs \
	MusicBrainz/Relation.cs \
	MusicBrainz/Release.cs \
	MusicBrainz/Track.cs \
	MusicBrainz/Utils.cs \
	MusicBrainz/XmlRequestEventArgs.cs

instdir = $(prefix)/lib/musicbrainz-sharp
inst_SCRIPTS = $(ASSEMBLY) $(ASSEMBLY).mdb

$(ASSEMBLY): $(SOURCES)
	$(MCS) -out:$@ -debug -target:library -unsafe $(REFERENCES) $(SOURCES)

all: $(ASSEMBLY)

EXTRA_DIST = $(SOURCES) MusicBrainz.csproj

CLEANFILES = $(ASSEMBLY) $(ASSEMBLY).mdb
DISTCLEANFILES = *.pidb
MAINTAINERCLEANFILES = Makefile Makefile.in
