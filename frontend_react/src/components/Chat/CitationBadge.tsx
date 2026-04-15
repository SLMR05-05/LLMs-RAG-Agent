import type { FC } from 'react';

interface CitationBadgeProps {
    number: number;
    sourceName: string;
    snippet: string;
    page?: number | null;
}

export const CitationBadge: FC<CitationBadgeProps> = ({
    number,
    sourceName,
    snippet,
    page,
}) => {
    return (
        <span className="group relative inline-flex align-baseline">
            <button
                type="button"
                className="ml-0.5 inline-flex h-5 w-5 items-center justify-center rounded-full border border-gray-200 bg-gray-100 text-[10px] font-semibold text-gray-700 shadow-sm transition-all duration-200 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 hover:shadow"
                aria-label={`Citation ${number}`}
            >
                {number}
            </button>

            <span className="pointer-events-none absolute left-1/2 top-full z-20 mt-2 hidden w-64 -translate-x-1/2 rounded-xl border border-gray-200 bg-white p-3 text-left text-xs text-gray-700 shadow-lg group-hover:block">
                <span className="mb-1 block font-semibold text-gray-900">
                    {sourceName}
                    {page ? ` · p.${page}` : ''}
                </span>
                <span className="block leading-relaxed text-gray-600">{snippet}</span>
            </span>
        </span>
    );
};

export default CitationBadge;
