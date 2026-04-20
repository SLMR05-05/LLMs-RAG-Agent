import type { FC } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useAppStore } from '../../store/useAppStore';

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
    const buttonRef = useRef<HTMLButtonElement | null>(null);
    const [open, setOpen] = useState(false);
    const [position, setPosition] = useState<{ top: number; left: number }>({ top: 0, left: 0 });

    const sources = useAppStore((state) => state.sources);
    const setActiveCitation = useAppStore((state) => state.setActiveCitation);
    const setSelectedSourceDetail = useAppStore((state) => state.setSelectedSourceDetail);

    const matchingSource = useMemo(() => {
        const normalizedName = sourceName.trim().toLowerCase();
        return (
            sources.find((source) => source.title.trim().toLowerCase() === normalizedName) ||
            sources.find((source) => source.title.toLowerCase().includes(normalizedName)) ||
            sources.find((source) => normalizedName.includes(source.title.toLowerCase())) ||
            null
        );
    }, [sourceName, sources]);

    useEffect(() => {
        if (!open || !buttonRef.current) {
            return;
        }

        const updatePosition = () => {
            if (!buttonRef.current) return;
            const rect = buttonRef.current.getBoundingClientRect();
            setPosition({
                top: rect.bottom + 10,
                left: rect.left + rect.width / 2,
            });
        };

        updatePosition();
        window.addEventListener('scroll', updatePosition, true);
        window.addEventListener('resize', updatePosition);
        return () => {
            window.removeEventListener('scroll', updatePosition, true);
            window.removeEventListener('resize', updatePosition);
        };
    }, [open]);

    const handleClick = () => {
        setActiveCitation({
            sourceId: matchingSource?.id,
            sourceName,
            snippet,
        });

        if (matchingSource) {
            setSelectedSourceDetail(matchingSource);
        }
    };

    return (
        <span className="inline-block align-baseline">
            <button
                ref={buttonRef}
                type="button"
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
                onFocus={() => setOpen(true)}
                onBlur={() => setOpen(false)}
                onClick={handleClick}
                className="ml-0.5 inline-flex h-5 w-5 items-center justify-center rounded-full border border-gray-200 bg-gray-100 text-[10px] font-semibold text-gray-700 shadow-sm transition-all duration-200 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700 hover:shadow"
                aria-label={`Citation ${number}`}
            >
                {number}
            </button>

            {open &&
                createPortal(
                    <div
                        className="pointer-events-none fixed z-[9999] w-80 -translate-x-1/2 rounded-xl bg-gray-900 px-3 py-2.5 text-left text-xs text-white shadow-2xl"
                        style={{ top: position.top, left: position.left }}
                    >
                        <div className="mb-1 font-semibold text-white">
                            {sourceName}
                            {page ? ` · p.${page}` : ''}
                        </div>
                        <div className="leading-relaxed text-gray-100">{snippet}</div>
                    </div>,
                    document.body
                )}
        </span>
    );
};

export default CitationBadge;
