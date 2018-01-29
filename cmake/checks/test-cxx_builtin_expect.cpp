int main()
{
    if(__builtin_expect(true, 1)) {
        return 0;
    } else {
        return 1;
    }
}
