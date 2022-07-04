

bool IsPowerOfTwo(int no)
{
	while (no >1)
	{
		if (no % 2 ==1)
			return false;
		no/=2;
	}

	
	return true;
}
