#include "history.h"

std::ostream& operator<<(std::ostream &out, const History &obj)
{
	out << "HISTORY:";
	for (auto beam : obj.history_) {
		out << beam.size() << " ";
	}

	return out;
}
