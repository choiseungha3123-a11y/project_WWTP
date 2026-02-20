package kr.kro.prjectwwtp.service;

import java.time.LocalDateTime;
import java.util.Optional;

import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import kr.kro.prjectwwtp.domain.Member;
import kr.kro.prjectwwtp.domain.Memo;
import kr.kro.prjectwwtp.domain.PageDTO;
import kr.kro.prjectwwtp.persistence.MemoRepository;
import lombok.RequiredArgsConstructor;

@Service
@RequiredArgsConstructor
public class MemoService {
	private final LogService logService;
	private final MemoRepository memoRepo;
	
	public Memo findByMemoNo(long memo_no) {
		Optional<Memo> opt = 	memoRepo.findById(memo_no);
		if(opt.isEmpty())
			return null;
		return opt.get();
	}
	
	public PageDTO<Memo> findByDisableMemberIsNull(Member member, Pageable pageable) {
		logService.addMemoLog(member, "list", pageable.getPageNumber(), pageable.getPageSize(), 0, null, null);
		return new PageDTO<>(memoRepo.findByDisableMemberIsNull(pageable));
	}
	
	public PageDTO<Memo> findByDisableMemberIsNotNull(Member member, Pageable pageable) {
		logService.addMemoLog(member, "oldlist", pageable.getPageNumber(), pageable.getPageSize(), 0, null, null);
		return new PageDTO<>(memoRepo.findByDisableMemberIsNotNull(pageable));
	}
	
	public void addMemo(Member member, String content, MultipartFile file) {
		Memo newMemo = null;
		byte[] imageData = null;
		try {
			if(file != null)
			{
				imageData = file.getBytes();
			}
			newMemo = Memo.builder()
					.content(content)
					.createMember(member)
					.fileName(file == null ? null : file.getOriginalFilename())
					.fileType(file == null ? null : file.getContentType())
					.imageData(imageData)
					.build();
			memoRepo.save(newMemo);
		}
		catch(Exception e) {
			logService.addErrorLog("MemoService.java", "addMemo()", e.getMessage());
		}finally {
			logService.addMemoLog(member, "create", 0, 0, newMemo.getMemoNo(), content, null);	
		}
	}
	
	public void modifyMemo(Member member, long memoNo, String content, MultipartFile file) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo modifyMemo = opt.get();
		byte[] imageData = null;
//		byte[] thumnailsData = null;
		try {
			logService.addMemoLog(member, "modify", 0, 0, memoNo, content, modifyMemo.getContent());
			modifyMemo.setContent(content);
			modifyMemo.setModifyMember(member);
			if(file != null) {
				modifyMemo.setFileName(file.getOriginalFilename());
				modifyMemo.setFileType(file.getContentType());
				modifyMemo.setImageData(imageData);
			}
			memoRepo.save(modifyMemo);
		}
		catch(Exception e) {
			logService.addErrorLog("MemoService.java", "modifyMemo()", e.getMessage());
		}finally {
		logService.addMemoLog(member, "modify", 0, 0, modifyMemo.getMemoNo(), content, null);	
		}
	}
	
	public void disableMemo(Member member, long memoNo) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo disableMemo = opt.get();
		logService.addMemoLog(member, "disable", 0, 0, memoNo, disableMemo.getContent(), null);
		disableMemo.setDisableMember(member);
		disableMemo.setDisableTime(LocalDateTime.now());
		memoRepo.save(disableMemo);
	}
	
	public void deleteMemo(Member member, long memoNo) throws Exception {
		Optional<Memo> opt = memoRepo.findByMemoNoAndDisableMemberIsNull(memoNo);
		if(opt.isEmpty())
			throw new Exception("memoNo가 올바르지 않습니다.");
		Memo deleteMemo = opt.get();
		logService.addMemoLog(member, "delete", 0, 0, memoNo, deleteMemo.getContent(), null);
		memoRepo.delete(deleteMemo);
	}

}
